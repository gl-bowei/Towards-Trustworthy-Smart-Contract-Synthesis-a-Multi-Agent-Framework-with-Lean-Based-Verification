import os
import re
import json
import time
from openai import OpenAI
from typing import Dict, List, Optional, Tuple

from core.config import get_simulation_agents
from core.logger import GLOBAL_LOGGER

# ==========================================
# Configuration
# ==========================================
API_KEY = os.getenv("LLM_API_KEY")
BASE_URL = os.getenv("LLM_BASE_URL")
DEFAULT_MODEL_NAME = "gpt-5"


class FatalLLMError(RuntimeError):
    """Raised when the configured LLM endpoint/model is unavailable for the run."""


def is_fatal_llm_error(error) -> bool:
    message = str(error).lower()
    fatal_tokens = [
        "no available channel",
        "model_not_found",
        "model not found",
        "invalid model",
        "unsupported model",
        "not support model",
        "does not exist",
        "invalid_api_key",
        "incorrect api key",
        "unauthorized",
        "permission_denied",
        "forbidden",
        "insufficient_quota",
        "quota_exceeded",
    ]
    status_code = getattr(error, "status_code", None)
    return status_code in {401, 403} or any(token in message for token in fatal_tokens)

# ==========================================
class BaseAgent:
    def __init__(self, model_name: str = None, api_key: str = None, base_url: str = None):
        self.api_key = api_key if api_key else (os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY"))
        self.base_url = base_url if base_url else os.getenv("LLM_BASE_URL")
        self.model_name = model_name if model_name else DEFAULT_MODEL_NAME
        self.request_timeout = float(os.getenv("LLM_REQUEST_TIMEOUT", "120"))

        if not self.api_key:
            print("Warning: No API Key provided for agent.")

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url) if self.api_key else None

    def _completion_kwargs(self) -> Dict:
        kwargs: Dict = {"timeout": self.request_timeout}
        model = (self.model_name or "").lower()
        if model.startswith("gpt-5"):
            max_tokens = int(os.getenv("LLM_MAX_COMPLETION_TOKENS", "12000"))
            reasoning_effort = os.getenv("LLM_REASONING_EFFORT", "low")
            kwargs["max_completion_tokens"] = max_tokens
            if reasoning_effort:
                kwargs["reasoning_effort"] = reasoning_effort
        else:
            kwargs["temperature"] = 0.2
        return kwargs

    def _clean_code(self, content: str) -> str:
        """
        Normalize model output by removing hidden-reasoning and Markdown markers.
        """
        if not content:
            return ""

        # 1. Remove <think>...</think>, including multiline blocks.
        # This keeps the parser compatible with models that emit reasoning tags.
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)

        # 2. Remove Markdown code-fence markers for parser compatibility.
        # Agents often return fenced blocks such as ```json or ```solidity.
        if "```" in content:
            content = re.sub(r"^```[a-zA-Z]*", "", content, flags=re.MULTILINE) # Opening fence
            content = re.sub(r"```$", "", content, flags=re.MULTILINE)         # Closing fence

        return content.strip()

    def _normalize_solidity_code(self, content: str) -> str:
        """Apply small, syntax-preserving Solidity cleanups to LLM output."""
        if not content:
            return ""
        # Solidity rejects `import { } from "...";`. Models also sometimes put
        # explanatory comments inside the braces; that is still an empty named
        # import, so normalize it to a side-effect import.
        def _empty_import_repl(match):
            body = match.group("body")
            body_without_comments = re.sub(r"/\*[\s\S]*?\*/|//[^\n]*", "", body).strip()
            if body_without_comments:
                return match.group(0)
            return f"import {match.group('path')};"

        content = re.sub(
            r'import\s*\{\s*(?P<body>[\s\S]*?)\s*\}\s*from\s*(?P<path>"[^"]+"|\'[^\']+\')\s*;',
            _empty_import_repl,
            content,
        )
        # A common LLM typo is `Type /*name*/ = expr;`, which Solidity parses as
        # a declaration without an identifier. Recover the intended variable name.
        content = re.sub(
            r"(?m)^(\s*(?:[A-Za-z_][A-Za-z0-9_]*\.)*[A-Za-z_][A-Za-z0-9_]*(?:\s+(?:memory|storage|calldata))?\s+)/\*\s*([A-Za-z_][A-Za-z0-9_]*)\s*\*/\s*=",
            r"\1\2 =",
            content,
        )
        # Solidity has no string-literal member suffix like `"N".bytes1`.
        # Normalize the recurring generated shorthand to the accepted cast form.
        content = re.sub(
            r'(?P<quote>["\'])(?P<char>[^"\'\\])(?P=quote)\s*\.\s*bytes1\b',
            r'bytes1("\g<char>")',
            content,
        )
        return content

    def _normalize_erc721_mock_approval(self, content: str) -> str:
        """Give generated ERC721 mocks the approval surface staking contracts expect."""
        source = content or ""
        if "MockERC721" not in source or "safeTransferFrom" not in source:
            return source

        if "interface IMockERC721" in source:
            def add_interface_methods(match: re.Match) -> str:
                body = match.group("body")
                if "setApprovalForAll" in body:
                    return match.group(0)
                body = body if body.endswith("\n") else body + "\n"
                insertion = (
                    "    function setApprovalForAll(address operator, bool approved) external;\n"
                    "    function isApprovedForAll(address owner, address operator) external view returns (bool);\n"
                )
                return f"{match.group('head')}{body}{insertion}{match.group('tail')}"

            source = re.sub(
                r"(?P<head>interface\s+IMockERC721\s*\{\n)(?P<body>[\s\S]*?)(?P<tail>\n\})",
                add_interface_methods,
                source,
                count=1,
            )

        contract_has_approval = bool(re.search(
            r"contract\s+MockERC721\b[\s\S]*?\bfunction\s+setApprovalForAll\s*\(",
            source,
        ))
        if not contract_has_approval:
            contract_has_approval_map = bool(re.search(
                r"contract\s+MockERC721\b[\s\S]*?\bisApprovedForAll\b",
                source,
            ))
            approval_state = ""
            if not contract_has_approval_map:
                approval_state = (
                    "    mapping(address => mapping(address => bool)) public isApprovedForAll;\n"
                    "    event ApprovalForAll(address indexed owner, address indexed operator, bool approved);\n\n"
                )
            elif "event ApprovalForAll" not in source:
                approval_state = (
                    "    event ApprovalForAll(address indexed owner, address indexed operator, bool approved);\n\n"
                )
            approval_block = (
                "\n"
                f"{approval_state}"
                "    function setApprovalForAll(address operator, bool approved) external {\n"
                "        isApprovedForAll[msg.sender][operator] = approved;\n"
                "        emit ApprovalForAll(msg.sender, operator, approved);\n"
                "    }\n"
            )
            source = re.sub(
                r"(contract\s+MockERC721\b[^{]*\{)",
                r"\1" + approval_block,
                source,
                count=1,
            )

        source = re.sub(
            r'require\s*\(\s*msg\.sender\s*==\s*from\s*,\s*"[^"]*"\s*\)\s*;',
            'require(msg.sender == from || isApprovedForAll[from][msg.sender], "not approved");',
            source,
        )
        source = re.sub(
            r'require\s*\(\s*msg\.sender\s*==\s*from\s*\|\|\s*msg\.sender\s*==\s*address\s*\(\s*this\s*\)\s*,\s*"[^"]*"\s*\)\s*;',
            'require(msg.sender == from || isApprovedForAll[from][msg.sender] || msg.sender == address(this), "no approval");',
            source,
        )
        return source

    def _payable_contract_names_from_source(self, source_code: str) -> List[str]:
        names: List[str] = []
        contract_pattern = re.compile(r"\bcontract\s+([A-Za-z_][A-Za-z0-9_]*)[^{}]*{")
        for match in contract_pattern.finditer(source_code or ""):
            name = match.group(1)
            open_brace = (source_code or "").find("{", match.end() - 1)
            if open_brace < 0:
                continue
            depth = 0
            end = None
            for idx in range(open_brace, len(source_code)):
                if source_code[idx] == "{":
                    depth += 1
                elif source_code[idx] == "}":
                    depth -= 1
                    if depth == 0:
                        end = idx
                        break
            body = source_code[open_brace + 1:end] if end is not None else ""
            if re.search(r"\breceive\s*\([^)]*\)\s*external\s+payable\b", body) or re.search(r"\bfallback\s*\([^)]*\)\s*external\s+payable\b", body):
                names.append(name)
        return names

    def _normalize_payable_target_casts(self, generated_code: str, target_code: str) -> str:
        """Normalize concrete casts for targets with payable receive/fallback."""
        code = generated_code or ""
        for contract_name in self._payable_contract_names_from_source(target_code):
            variable_cast = re.compile(
                rf"\b{re.escape(contract_name)}\(\s*(?!payable\s*\()(?P<arg>[A-Za-z_][A-Za-z0-9_]*)\s*\)"
            )
            address_cast = re.compile(
                rf"\b{re.escape(contract_name)}\(\s*address\s*\(\s*(?P<arg>[A-Za-z_][A-Za-z0-9_]*)\s*\)\s*\)"
            )

            def _cast_variable(match):
                prefix = code[max(0, match.start() - 8):match.start()]
                if re.search(r"\bnew\s+$", prefix):
                    return match.group(0)
                return f"{contract_name}(payable({match.group('arg')}))"

            def _cast_address(match):
                prefix = code[max(0, match.start() - 8):match.start()]
                if re.search(r"\bnew\s+$", prefix):
                    return match.group(0)
                return f"{contract_name}(payable(address({match.group('arg')})))"

            code = address_cast.sub(_cast_address, code)
            code = variable_cast.sub(_cast_variable, code)
        return code

    def _strip_zero_value_call_options(self, generated_code: str) -> str:
        """Remove no-op `{value: 0}` call options that break non-payable calls."""
        return re.sub(
            r"(\.[A-Za-z_][A-Za-z0-9_]*)\s*\{\s*value\s*:\s*(?:0|uint256\s*\(\s*0\s*\)|0\s+(?:wei|ether))\s*\}\s*\(",
            r"\1(",
            generated_code or "",
        )

    def _query_llm(self, system_prompt: str, user_prompt: str) -> str:
        max_attempts = int(os.getenv("LLM_MAX_RETRIES", "4"))
        retry_delay = float(os.getenv("LLM_RETRY_BASE_DELAY", "5"))
        for attempt in range(1, max_attempts + 1):
            try:
                if self.client is None:
                    raise RuntimeError("No LLM API key configured. Set LLM_API_KEY or OPENAI_API_KEY.")
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    **self._completion_kwargs(),
                )
                raw_content = response.choices[0].message.content or ""

                if os.getenv("LEVER_PRINT_RAW_LLM_OUTPUT", "0").strip().lower() in {
                    "1", "true", "yes", "on"
                }:
                    print(f"\n{'='*20} [RAW LLM OUTPUT START: {self.__class__.__name__}] {'='*20}")
                    print(raw_content)
                    print(f"{'='*20} [RAW LLM OUTPUT END] {'='*20}\n")

                GLOBAL_LOGGER.log_agent(self.__class__.__name__, user_prompt, raw_content)

                return self._clean_code(raw_content)

            except Exception as e:
                error_msg = f"LLM Error ({self.__class__.__name__}): {e}"
                print(error_msg)
                GLOBAL_LOGGER.log_system("LLM_ERROR", error_msg)
                if is_fatal_llm_error(e):
                    raise FatalLLMError(error_msg) from e
                retryable = any(token in str(e).lower() for token in [
                    "429",
                    "rate_limit",
                    "proxy_rate_limit",
                    "upstream",
                    "connection error",
                    "timeout",
                ])
                if retryable and attempt < max_attempts:
                    sleep_s = retry_delay * attempt
                    print(f"   Retrying LLM call in {sleep_s:.1f}s ({attempt}/{max_attempts})...")
                    time.sleep(sleep_s)
                    continue
                return ""


class SemanticObligationExtractor:
    """
    Deterministically extracts requirement-derived semantic obligations.

    These obligations are intentionally kept separate from the fixed property
    taxonomy: they are inferred from the prompt/business workflow and should be
    reported as lower-confidence semantic checks in experiment analysis.
    """

    def extract(self, solidity_code: str, user_targets: List[str]) -> List[Dict]:
        requirement_text = "\n".join(user_targets or [])
        combined = f"{requirement_text}\n{solidity_code or ''}"
        lowered = combined.lower()
        obligations: List[Dict] = []

        def has_any(words: List[str]) -> bool:
            return any(word in lowered for word in words)

        def add_obligation(
            obligation_id: str,
            name: str,
            text: str,
            confidence: str,
            reason: str,
            suggested_checks: List[str],
        ):
            if any(o["id"] == obligation_id for o in obligations):
                return
            obligations.append({
                "id": obligation_id,
                "name": name,
                "text": text,
                "confidence": confidence,
                "reason": reason,
                "suggested_checks": suggested_checks,
            })

        mint_terms = ["mint", "mints", "minting", "issuance", "issue"]
        distribute_terms = [
            "distribute", "distribution", "airdrop", "reward", "team",
            "percentage", "percentages", "allocation", "allocate",
        ]
        cap_terms = [
            "limited", "limit", "cap", "maximum", "max", "per wallet",
            "per-address", "per address", "whitelist", "presale", "allowlist",
        ]
        proxy_terms = [
            "proxy", "clone", "minimal proxy", "eip-1167", "factory",
            "mastercopy", "master copy", "delegatecall",
        ]

        liability_signals = [
            "deposit", "withdraw", "claim", "redeem", "escrow", "vault",
            "collateral", "fund", "funds", "share", "shares", "msg.value",
            "payable", "ether", "eth",
        ]
        if has_any(liability_signals) and has_any(["balance", "balances", "claim", "withdraw", "share", "fund", "deposit"]):
            add_obligation(
                "SEMANTIC-LIABILITY-SOLVENCY",
                "Liability solvency",
                "User/team/accounting liabilities should never exceed assets that are actually escrowed or otherwise available to satisfy them.",
                "medium",
                "Requirement/code mentions monetary balances, claims, shares, funds, or withdrawals.",
                [
                    "Check sum of exposed claimable balances for tracked users is <= contract ETH/token balance when getters expose enough state.",
                    "After a successful withdrawal, the matching claim should decrease by the withdrawn amount.",
                    "Repeated settlement/distribution should not duplicate claimable liabilities without consuming source assets.",
                ],
            )

        if has_any(distribute_terms) and has_any(["withdraw", "claim", "balance", "balances", "fund", "funds"]):
            add_obligation(
                "SEMANTIC-REPLAY-CONSUMPTION",
                "Replay/consumption semantics",
                "Settlement, distribution, airdrop, claim, and withdrawal flows should be one-shot or monotonic: repeating a successful effect must consume entitlement/assets or be idempotent.",
                "medium",
                "Requirement/code mentions distribution-like flows together with balances or withdrawals.",
                [
                    "Exercise repeated successful calls to distribution/claim/withdraw functions from valid states.",
                    "Compare claim balances, total distributed counters, and contract assets before/after repeated calls.",
                    "Treat idempotent no-op repeats as safe, but duplicated credit without source debit as unsafe.",
                ],
            )

        if has_any(cap_terms) and has_any(mint_terms + ["buy", "purchase", "sale", "tokens"]):
            add_obligation(
                "SEMANTIC-PER-USER-CAP",
                "Per-user cap/allowlist semantics",
                "When the requirement says a presale/whitelist/limited mint is capped, the cap must hold across repeated successful calls by the same account.",
                "medium",
                "Requirement/code mentions limits, caps, whitelist/allowlist, or presale together with minting/purchase.",
                [
                    "Drive repeated mint/buy calls by the same allowed user after valid setup.",
                    "Track per-user minted/purchased amount independently from totalSupply.",
                    "Do not accept totalSupply-only checks as sufficient for per-user requirements.",
                ],
            )

        if has_any(["deposit", "deposits"]) and has_any(["withdraw", "withdraws", "redeem", "burn"]) and has_any(mint_terms + ["token", "tokens", "wrapped", "receipt"]):
            add_obligation(
                "SEMANTIC-ASSET-BACKING",
                "Deposit/mint backing",
                "If deposits mint or credit receipt tokens/shares, successful minting must correspond to received/locked backing assets, and withdrawals must return or release the backing.",
                "medium",
                "Requirement/code mentions deposit and withdrawal/redeem together with minted tokens or shares.",
                [
                    "Before/after deposit, check that user token/share credit increases only when ETH/token backing also enters custody.",
                    "Before/after withdraw/redeem, check that receipt balance decreases and backing leaves custody to the user.",
                    "Flag free minting paths where amount parameters create credit without asset transfer or msg.value.",
                ],
            )

        if has_any(proxy_terms):
            add_obligation(
                "SEMANTIC-PROXY-USABILITY",
                "Factory/proxy usability",
                "Factory-created proxy/clone instances should have runtime code, initialize correctly, and support the expected delegated behavior and storage isolation.",
                "medium",
                "Requirement/code mentions factory, proxy, clone, master copy, or delegatecall.",
                [
                    "After create/clone/deploy, assert the returned address has nonzero code.",
                    "Call at least one expected method on the created instance, not only the factory getter.",
                    "Check independent proxy instances do not accidentally share user-facing storage.",
                ],
            )

        if has_any(["state", "phase", "start", "finish", "open", "close", "initialize", "initialized", "init"]):
            add_obligation(
                "SEMANTIC-POSITIVE-PATH",
                "Positive-path reachability",
                "The main lifecycle described by the requirement should be reachable by authorized actors before adversarial or invariant checks judge the contract.",
                "low",
                "Requirement/code mentions lifecycle states, initialization, opening/closing, or setup.",
                [
                    "In fuzz setup, execute required owner/admin initialization and phase-opening calls when exposed.",
                    "Attack scripts should distinguish invalid pre-state reverts from security-preserving reverts.",
                    "Semantic conclusions should be marked low confidence if the positive path cannot be reached.",
                ],
            )

        return obligations


def format_semantic_obligations(obligations: List[Dict], include_low_confidence: bool = True) -> List[str]:
    formatted = []
    for obligation in obligations or []:
        if not include_low_confidence and obligation.get("confidence") == "low":
            continue
        checks = "; ".join(obligation.get("suggested_checks", [])[:3])
        formatted.append(
            f"[{obligation.get('id')}] {obligation.get('name')}: "
            f"{obligation.get('text')} "
            f"{{Confidence: {obligation.get('confidence', 'medium')}; "
            f"Reason: {obligation.get('reason', '')}; Suggested Checks: {checks}}}"
        )
    return formatted

# ==========================================
# 1. Property Selector
# ==========================================
class PropertySelector(BaseAgent):
    def __init__(self, json_path="property.json", model_name: str = None, api_key: str = None, base_url: str = None):
        super().__init__(model_name=model_name, api_key=api_key, base_url=base_url)
        self.json_path = json_path
        self.property_map = {} # Map property IDs to their complete objects.
        self.knowledge_base_str = ""
        self._load_knowledge_base()

    def _load_knowledge_base(self):
        """Load and flatten the JSON for prompting while building an index."""
        if not os.path.exists(self.json_path):
            print(f"Warning: {self.json_path} not found. Using empty base.")
            return

        try:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            kb_lines = []
            self.property_map = {}

            # Traverse Level -> Category -> Properties.
            for lvl_name, lvl_data in data.items():
                kb_lines.append(f"\n[{lvl_name}]")
                kb_lines.append(f"Description: {lvl_data.get('description', '')}")

                categories = lvl_data.get("categories", {})
                for cat_name, cat_data in categories.items():
                    kb_lines.append(f"  > {cat_name}: {cat_data.get('description', '')}")

                    for prop in cat_data.get("properties", []):
                        pid = prop.get("id")
                        pname = prop.get("name")
                        pdesc = prop.get("description")
                        # Store the object for later lookup.
                        self.property_map[pid] = prop
                        # Add the property to the prompt text.
                        kb_lines.append(f"    - ID: {pid} | Name: {pname}")
                        kb_lines.append(f"      Desc: {pdesc}")

            self.knowledge_base_str = "\n".join(kb_lines)

        except Exception as e:
            print(f"Error loading property.json: {e}")

    def select_properties(
        self,
        solidity_code: str,
        user_targets: List[str],
        existing_properties: Optional[List[str]] = None,
    ) -> List[str]:
        existing_properties = existing_properties or []
        target_text = "\n".join([f"- [USER TARGET] {t}" for t in user_targets])
        preserved_text = "\n".join([f"- {p}" for p in existing_properties]) if existing_properties else "None"

        system_prompt = f"""
        You are a Security Requirement Engineer.
        Your goal is to select the most relevant verification properties from the Knowledge Base for a specific Smart Contract.

        [KNOWLEDGE BASE]
        {self.knowledge_base_str}

        [TASK]
        1. Analyze the input Solidity code.
        2. Identify which properties from the Knowledge Base are relevant (e.g., if there is math, select Arithmetic Safety; if there is a loop, select Gas Safety).
        3. Apply an applicability filter before selecting:
           - Select ERC standard compliance ONLY when the contract explicitly claims or implements ERC20/ERC721/ERC1155 behavior
             (inheritance, interface names, standard function set, or user requirement). A custom "NFT-like" accounting contract is NOT enough.
           - Select role-separation properties only for privileged/admin-sensitive functions or explicit access-control requirements.
           - Select solvency/withdrawal properties only when the contract tracks monetary balances, claims, deposits, shares, or ETH liabilities.
           - Do NOT select a property just because it is generally good practice; it must be testable from the target interface or requirements.
        4. **CRITICAL**: Return ONLY the list of Property IDs (e.g., "L1.1-Overflow").
        5. Also include the User Defined Targets as strict constraints.

        [OUTPUT FORMAT]
        - Output a pure list of IDs, one per line.
        - Example:
          L1.1-Overflow
          L1.3-Reentrancy
          L2.3-Solvency
        """

        user_prompt = f"""
        [INPUT SOLIDITY]
        {solidity_code}

        [USER DEFINED TARGETS]
        {target_text}

        [PRESERVED PROPERTIES]
        {preserved_text}

        [OUTPUT]
        List relevant Property IDs only.
        """

        print("   PropertySelector is identifying verification goals from JSON...")
        response = self._query_llm(system_prompt, user_prompt)

        # Parse returned IDs and recover complete records, including templates.
        final_properties = []

        # 1. Preserve existing free-form text properties.
        for p in existing_properties:
            final_properties.append(p)

        # 2. Parse the newly selected IDs.
        lines = response.split('\n')
        for line in lines:
            clean_id = line.strip().replace("- ", "").replace("*", "").strip()

            if clean_id in self.property_map:
                if not self._is_property_applicable(clean_id, solidity_code, user_targets):
                    continue
                # Build a detailed description, including the template, for known IDs.
                prop_data = self.property_map[clean_id]
                # Format: [ID] Name: Description [Invariant: ...]
                rich_desc = (
                    f"[{clean_id}] {prop_data['name']}: {prop_data['description']} "
                    f"{{Invariant Template: {prop_data.get('invariant_template', 'N/A')}}}"
                )
                final_properties.append(rich_desc)
            elif len(clean_id) > 5 and "L" not in clean_id:
                # Keep custom targets and responses that do not follow the ID format.
                final_properties.append(clean_id)

        GLOBAL_LOGGER.log_system("SELECTED_PROPERTIES", str(final_properties))
        return final_properties

    def _is_property_applicable(self, prop_id: str, solidity_code: str, user_targets: List[str]) -> bool:
        text = f"{solidity_code}\n" + "\n".join(user_targets or [])
        lowered = text.lower()

        if prop_id == "L1.4-ERCCompliance":
            erc_signals = [
                "erc20", "ierc20", "erc721", "ierc721", "erc1155", "ierc1155",
                "safeTransferFrom", "transferFrom", "approve(", "allowance(",
                "ownerOf(", "balanceOf(", "tokenURI(", "supportsInterface(",
                "totalSupply(", "transfer(address", "transfer(",
            ]
            return any(signal.lower() in lowered for signal in erc_signals)

        if prop_id in {"L2.2-RoleSeparation"}:
            role_signals = [
                "owner", "admin", "role", "onlyowner", "onlyrole", "accesscontrol",
                "setstate", "pause", "upgrade", "withdraw", "sendfunds",
            ]
            return any(signal in lowered for signal in role_signals)

        if prop_id in {"L2.3-Solvency", "L3.1-Withdrawal", "L3.2-PullPattern", "L3.2-ForcedEther"}:
            money_signals = [
                "payable", "address(this).balance", "balance", "balances", "deposit",
                "withdraw", "claim", "share", "fund", "ether", "msg.value",
            ]
            return any(signal in lowered for signal in money_signals)

        return True

# ==========================================
# 2. Lean Formalizer
# ==========================================
class LeanFormalizer(BaseAgent):
    """
    Translates Solidity to Lean 4 Definitions & Theorem Statements.
    Produces theorem statements with `sorry` placeholders; proof generation is
    handled by LeanProver.
    """
    def __init__(self, model_name: str = None, api_key: str = None, base_url: str = None):
        # Forward the API key and base URL to the base class.
        super().__init__(model_name=model_name, api_key=api_key, base_url=base_url)
    def formalize_definitions(
        self,
        solidity_code: str,
        properties: List[str],
        error_feedback: str = "",
        theorem_budget: int = 0,
    ) -> str:

        props_text = "\n".join([f"{i+1}. {p}" for i, p in enumerate(properties)])
        budget = max(0, int(theorem_budget or 0))
        theorem_budget_section = ""
        theorem_task_line = f"Put all {len(properties)} properties (and any helpers) under theorems."
        if budget > 0:
            theorem_budget_section = f"""
        [THEOREM BUDGET - PAPER-ALIGNED MODE]
        Generate AT MOST {budget} theorem declarations in the `/- [SECTION: THEOREMS] -/` block.
        Treat these as CORE theorem obligations for paper-facing statistics.
        - Prefer one canonical theorem per selected property.
        - If there are more selected properties than the budget, prioritize the
          most security-critical and contract-specific obligations: access
          control, solvency/accounting, lifecycle/state-machine, external-call
          ordering, and representative guard checks.
        - Do not split one property into separate micro-theorems for every event,
          return-shape, and zero guard unless those are the main selected property.
        - Helper lemmas should be avoided; if emitted as `theorem`, they count
          toward the {budget} theorem budget. Prefer helper `def`s instead.
        - The goal is roughly paper-style obligation count, not exhaustive
          engineering-debug theorem coverage.
            """
            theorem_task_line = (
                f"Put at most {budget} CORE theorem obligations under theorems, "
                "choosing representative obligations for the selected properties."
            )

        scaffold_section = ""
        if os.getenv("LEVER_LEAN_SANITY_SCAFFOLD", "0").lower() in {"1", "true", "yes"}:
            scaffold_section = """
        [LEAN SANITY SCAFFOLD MODE]
        This run is a high-value sanity check, not exhaustive theorem mining.
        Generate a small, proof-friendly model:
        - Add `set_option autoImplicit false` immediately after `import Lean`.
        - Every theorem variable must be explicitly bound. In particular, if a
          theorem mentions `out`, bind `(out : TransitionOutput)` explicitly.
        - Never use metavariable holes or inferred placeholders: forbidden forms
          include `?out`, `?e`, `?_`, `events := _`, `transfers := _`, and
          omitted record fields.
        - Every theorem placeholder must be exactly `:= sorry`. Do not output
          `:= := sorry`, `:= by sorry`, or theorem proof bodies.
        - Do not use parser-sensitive binder names such as `from`, `to`, `type`,
          `match`, `where`, `by`, or `end`; use descriptive names such as
          `fromAddr`, `toAddr`, `assetKind`, `newOwner`, or `targetAddr`.
          This applies to theorem binders, function parameters, lambda binders,
          and match-branch binders. For example, avoid `fun at => ...`; write
          `fun assetKind => ...`.
        - Never make a theorem conclude `True`, `False`, `x = x`, `P -> True`,
          `P ∧ True`, or `P ∨ True`. Every theorem conclusion must mention a
          concrete state field, event list, transfer list, or failure result.
        - Prefer local transition facts over global reachability facts. Do not
          put `ReachC` in theorem statements in this mode unless the selected
          property explicitly requires reachability.
        - You may define helper `def`s or predicates to expose branch conditions,
          but do not create helper `theorem`s unless they are within the theorem
          budget and security-relevant.
        - For authorization and role properties, prefer exact guard theorems with
          the full guard prefix made explicit:
          `guard_prefix -> transition ... = CallResult.failure Error.foo`.
          If a failure branch is not the first failure branch in the relevant
          transition/helper, the theorem must include every earlier branch
          precondition needed to reach that branch. For example, do not assert a
          later zero-address, cap, allowance, phase, or external-call failure
          without first ruling out earlier sender, phase, balance, or role
          failures.
        - For accounting, custody, backing, cap, replay, or consumption
          properties, split broad obligations into local success-shape theorems
          with all branch preconditions explicit. Each theorem should assert one
          primary state effect or one primary failure condition.
        - For success-path obligations, bind `(out : TransitionOutput)` and use
          the shape `preconditions -> transition ... = CallResult.success out ->
          one concrete postcondition`. Avoid existential wrappers when the same
          property can be stated with an explicit `out` argument.
        - For map/function updates, define small setter/getter helper `def`s and
          write theorem statements that mention one key at a time. Include
          distinct-key assumptions explicitly when a property depends on them.
        - If a transition loops over a list, extract the loop into a top-level
          helper `def` before the public transition. Prefer theorems about a
          single guard, one-step field preservation, or a one-element/success
          shape over a broad recursive invariant.
          Do not define repeated local recursive helpers named `loop`, `go`, or
          `process` inside multiple branches of the same dispatcher; Lean can
          elaborate them to colliding internal declaration names. Use top-level
          helpers with unique action-specific names instead.
        - If you define `ReachC`, use exactly this conservative shape: a binary
          state-to-state relation with `refl` and one generic `step` constructor
          over the public input dispatcher. Do not put trace lists, context
          lists, action lists, `[]`, `::`, or `++` in the ReachC indices. Do not
          create one ReachC constructor per public action. The step constructor
          must bind `(out : TransitionOutput)` and use
          `dispatcher currentState ctx input = CallResult.success out`, then
          conclude reachability of `out.state`.
        - Avoid theorem conclusions with large conjunctions. If a candidate
          property has several independent consequences, emit separate core
          obligations for the separate field updates or guards.
        - For multi-step workflow properties, prefer theorem statements about
          the nearest modeled transition/helper that performs the relevant state
          change. Do not force an entire public workflow into one theorem when a
          smaller local transition theorem captures the same safety obligation.
        - Avoid exact event-list or transfer-list postconditions unless the event
          is the core property. State storage/accounting/role effects first.
            """

        feedback_section = ""
        if error_feedback:
            feedback_section = f"""
            [PREVIOUS COMPILATION ERRORS]
            The previous Lean code failed to compile. Fix these specific errors:
            {error_feedback}
            """

        system_prompt = f"""
        You are a Formal Verification Engineer expert in Lean 4.
        {feedback_section}
        Your task is to translate Solidity code into a Lean 4 State Machine.
        {theorem_budget_section}
        {scaffold_section}

        [CRITICAL OUTPUT STRUCTURE - STRICT ENFORCEMENT]
        To ensure the prover agent works correctly, you MUST structure your output using specific section markers.

        Structure your code EXACTLY as follows:

        /- [SECTION: DEFINITIONS] -/
        import Lean
        set_option autoImplicit false
        // ... Define structure State, Context, Error ...
        // ... Implement all helper functions and transitions ...
        // ... (Do NOT put any theorems here) ...

        /- [SECTION: THEOREMS] -/
        // ... Define theorems for properties here ...
        // ... If a property requires multiple theorems, list them all here ...

        [STRICT MODELING STYLE GUIDE]
        1. **Standard Data Structures**:
           - `structure State`: Contract storage. Use `Nat` for uint.
           - `structure Context`: `sender`, `value`, `timestamp`.
           - `structure Event`: emitted events or externally visible observations.
           - `structure Transfer`: ETH/ERC20/ERC721 movement reified as data.
           - `structure TransitionOutput`: updated `State`, emitted `events`, and `transfers`.
           - `inductive Error`: concrete failure reasons.
           - `inductive CallResult`: `| success (out : TransitionOutput) | failure (err : Error)`.
           - Order matters: define `State` before `TransitionOutput`; define
             `TransitionOutput` before `CallResult`; define all helper functions
             before any theorem that references them.
           - Every custom `inductive` used in equality tests, `if h : x = y`,
             list equality, event equality, or transfer equality MUST include
             `deriving DecidableEq`. This includes state enums, `Error`, `Event`,
             `Transfer`, and action/input enums.
           - Do NOT derive equality or `Repr` for `State` or any structure with
             function fields such as `Address → Nat`.
           - Do NOT derive `DecidableEq` for `TransitionOutput`, `CallResult`, or
             any wrapper that contains `TransitionOutput` when `State` has
             function-valued fields. Equality for these types forces equality for
             functions and commonly makes the definitions fail to compile.
             Prefer matching on `CallResult.success out` / `CallResult.failure e`
             in theorem statements instead of testing whole-output equality.
           - If an `inductive` with `deriving DecidableEq` has a constructor
             argument whose type is a custom `structure`, that structure must
             also support `DecidableEq`, unless it contains function fields. If
             the structure has function fields or you are unsure, do NOT derive
             `DecidableEq` for the enclosing action/input inductive. The
             transition `match` does not require input equality.

        2. **Paper-Aligned Semantic Model**:
           - Model the Solidity contract as `(ΣC, Γ, IC, ΥC)`:
             `ΣC` is `State`, `Γ` is `Context`, `IC` is the public-call input type,
             and `ΥC` is the transition relation/function.
           - If reachability is needed, define a predicate named `ReachC` over
             repeated valid transitions. Prefer the stable binary shape
             `inductive ReachC : State -> State -> Prop` with `refl` and one
             generic `step` constructor that binds `(out : TransitionOutput)`,
             assumes the public dispatcher returns `CallResult.success out`,
             and concludes reachability of `out.state`. Do not index ReachC by
             trace lists unless explicitly required by a property.
           - Reify side effects. Do NOT hide balance changes, token transfers, event
             emissions, role changes, or external-call results inside comments.
           - Theorems should quantify over reachable states when appropriate, e.g.
             properties of all `s` such that `ReachC init s`.

        3. **Conservative Lean Subset for Transitions**:
           - Keep transition functions first-order and executable.
           - Prefer ordinary `if condition then ... else` over dependent
             proof-binding when the proof name is unused. If you write
             `if h : condition then`, make sure Lean can synthesize a
             `Decidable condition`; for custom enum equalities this requires
             `deriving DecidableEq`.
           - Return only `CallResult.success` or `CallResult.failure`.
           - Avoid mutually recursive definitions, typeclasses, coercions,
             namespaces, custom notation, `where` blocks, and dependent records.
           - Avoid referencing variables introduced only inside an `if` branch
             from outside that branch.

        4. **Logic Pitfalls to AVOID**:
           - **NO `deriving Repr`** on structures with functions.
           - **NO `deriving DecidableEq`** on structures that contain function
             fields such as `balances : Address → Nat`, `stakes : Address → Nat`,
             or nested function-valued maps.
           - For plain configuration/value structures that contain only
             first-order comparable fields such as `Nat`, `Bool`, addresses,
             enums, strings, or lists of comparable values, either add
             `deriving DecidableEq` when they are used inside event/input equality,
             or avoid deriving equality for the event/input type that contains
             them. Never leave a derived equality depending on a structure that
             has no decidable equality instance.
           - **No Mathlib**: Use standard Lean 4 types only.
           - Do not use Unicode syntax unless it is already common Lean syntax
             in simple comparisons (`≤`, `≥`, `≠` are allowed, but ASCII
             alternatives are preferred if they keep the term simple).
           - Avoid fragile List APIs that may not exist in the current Lean
             environment, including `List.set!`, `List.get!`, bare `.get!`, and
             complex pipeline field access like `x |>.field`. Define small helper
             functions such as `setNth`, `get?`, or pattern-match on `List.get?`
             instead.
           - Avoid parser-sensitive pipeline field access in theorem statements,
             proof terms, and helper expressions. Do not write
             `(f x |>.field)` or `let y := f x |>.field`; instead bind the
             intermediate value first or use ordinary parenthesized field access:
             `let u := f x; u.field`.
           - Avoid Lean reserved words and parser-sensitive names for identifiers
             or structure fields. In particular, do NOT use fields named `from`,
             `to`, `type`, `match`, `where`, `by`, or `end`. Use names like
             `fromAddr`, `toAddr`, `assetKind`, `eventKind`.
           - Also avoid Lean command/parser-sensitive names for functions and
             constructors, including `initialize`, `constructor`, `section`,
             `namespace`, `open`, `closed`, `theorem`, and `example`. Use names
             like `initContract`, `initTransition`, `saleOpen`, and
             `saleClosed`.
           - Avoid wrapping the output in namespaces unless absolutely necessary.
             If you use a namespace, open it only once and close it only once; do
             not emit nested duplicate namespaces around the theorem section.
           - Do NOT state global invariants over arbitrary `init` states unless the
             theorem also assumes a well-formed initial state. For reachability
             theorems, either define `ReachC` from the constructor/initial-state
             output, or add explicit assumptions such as initialized, fixed
             supply, owner initial balance, zero initial distribution, and any
             accounting relation needed by the invariant.
           - Reachability invariants over `ReachC init s` must include a well-formed
             initial-state assumption strong enough for the base case. Because
             `ReachC.refl` makes `init` reachable from itself, claims such as
             `s.stakeholders.length ≤ s.maxStakeholders` require an assumption
             like `init.stakeholders.length ≤ init.maxStakeholders`.
           - Do NOT claim token/cap/solvency invariants that are stronger than the
             modeled transition system. If tokens can be transferred back, minted,
             burned, or sold repeatedly, encode the required cap state or add the
             missing preconditions; otherwise generate a local pre/post theorem
             instead of a false global invariant.
           - Respect guard order in transition functions. If a theorem states that
             a later guard returns a specific failure reason, include hypotheses
             that all earlier guards pass. Example: a `nonStandardToken` failure
             theorem must assume earlier reentrancy, zero-amount, balance, and cap
             guards do not fire first. Otherwise prefer a success-implies or
             guarded-local theorem.
           - This guard-order rule also applies to gas/cap/loop-bound failures.
             A theorem like `stakeholders.length ≥ maxStakeholders → createStake = failure gasDoS`
             is false unless it also assumes earlier guards pass, such as
             `entered = false` and `stake > 0`.
             Likewise, an airdrop/list loop-bound theorem must assume the contract
             is not already in an earlier-failing phase and that length-mismatch
             checks pass before claiming the loop-bound error:
             `state ≠ distribution → recipients.length = quantities.length →
              recipients.length > maxIterations → airdrop ... = failure gasDoS`.
           - For exact failure-code theorems, list the full guard prefix needed
             to reach that branch. If a function checks initialization, owner,
             sale phase, nonzero input, balance, allowance, or cap before the
             branch you care about, include those earlier guards as hypotheses.
             Otherwise state a weaker theorem such as "success preserves/updates
             field X" instead of "condition Y returns Error.Z".
           - For success-path theorems about transfer, buy, sell, claim, mint, or
             start/execute functions, include the full earlier guard prefix when
             the conclusion depends on a later branch: nonzero sender/from,
             nonzero recipient/to, nonzero amount, initialized/open phase,
             owner/keeper/registered-user checks, sufficient balance/allowance,
             cap/limit checks, cooldown checks, and not-paused/not-reentrant
             flags. If these guards would make the theorem long, choose a local
             theorem about the immediate branch condition or the exact returned
             output instead of a broad postcondition.
           - Do not assert arithmetic postconditions that need an invariant unless
             the theorem includes that invariant. For example,
             `totalStakes - amount + amount = totalStakes` needs
             `amount ≤ totalStakes`; checking `amount ≤ userStake` is not enough
             unless a prior invariant connects `userStake` to `totalStakes`.
           - Do not invent placeholder arithmetic, tautological hypotheses, or
             catch-all event alternatives just to make a theorem syntactically
             broad. Avoid expressions such as division by zero, irrelevant
             `true` facts, or event branches that do not correspond to a real
             transition branch. Prefer a smaller guarded theorem with an
             interpretable postcondition.
           - For transfers, handle self-transfer (`sender = recipient`) explicitly.
             Avoid sum-of-two-balances equalities that double-count the same
             account unless the theorem has a `sender ≠ recipient` hypothesis.
           - For any theorem about updating two map keys, include a disjointness
             hypothesis if the keys may be equal. Examples:
             `ctx.sender ≠ toAddr` for transfer sender/recipient postconditions,
             `ctx.sender ≠ s.owner` for buy/sale owner-buyer postconditions, and
             pairwise distinctness for batched recipients.
             This applies to BOTH sides of the write. A theorem about the debited
             sender/from balance also needs `ctx.sender ≠ toAddr` or
             `fromAddr ≠ toAddr` when the implementation later credits `toAddr`;
             otherwise self-transfer can overwrite the debit.
           - If the contract intentionally treats self-transfer or owner-buying as
             a no-op/special case, write a separate theorem for that special case
             or weaken the theorem to avoid exact balance deltas.
           - Division and precision theorems must include the exact denominator
             and nonzero/range assumptions needed by the modeled code. Do not
             claim "no precision loss" as equality unless the remainder is
             explicitly assumed to be zero; otherwise prove the actual rounded
             expression used by the transition.
           - Loop/gas-bound theorems must either be local success properties over
             the modeled bounded list or include the invariant that the list is
             within the bound before the transition.
           - Recursive mint/distribution/list-processing helpers are expensive to
             prove directly. Prefer theorem statements over the public transition
             that expose simple local consequences of success. Only state a loop
             count, supply cap, or exact transfer-count theorem when the model has
             a small helper whose recursion exposes that fact and the theorem
             includes the necessary loop invariants.
           - For batched accounting over recipient/team lists, do not state an
             aggregate liability or aggregate balance equality unless the theorem
             includes the list invariants needed by the loop: matching lengths,
             `List.Nodup`/pairwise distinct recipient addresses, and any
             percentage-sum or per-entry bound assumptions. If duplicates are
             allowed or not modeled, prefer a local per-recipient/per-index
             postcondition or state the exact folded update expression.
           - For list/map transfer-count properties, avoid proving a theorem that
             requires Lean to discover `List.map.length` or constructor equality
             through a large transition. Either define a small helper and a direct
             theorem about that helper, or choose an event/state theorem that
             avoids recursive list-length reasoning.
           - For reentrancy-lock or "unlock" postconditions, account for early
             success returns/no-op branches. A theorem such as
             `out.state.inSwap = false` is false if the function can return the
             original state successfully while `s.inSwap = true`; add hypotheses
             excluding the early-return branch, assume the initial flag value, or
             state a weaker branch-local theorem.
           - A theorem should represent a check that follows from the model. If a
             property is only a desired safety goal but the code/model lacks the
             guard needed to imply it, state the guarded/local property rather than
             silently strengthening the contract semantics.
           - Do NOT use `True`, `False`, `∨ True`, `True ∨ ...`, or placeholder
             theorems such as `safeERC20_placeholder`. For success-only properties,
             prefer implication form:
             `function s ctx args = CallResult.success out → meaningful_postcondition`.
             Do not encode a failed-call branch as `| failure _ => True` unless the
             property is explicitly about both success and failure behavior.
           - Be careful with theorem statements that use `x ∈ out.events`,
             `List.contains`, or membership over custom inductives. If you use
             membership/contains, ensure the involved event/transfer inductives
             derive `DecidableEq`, and never derive equality for structures with
             function fields. When in doubt, state event/transfer properties as
             concrete list equality, e.g. `out.events = [Event.foo ...]`, or as an
             existential over the whole list shape rather than relying on
             membership search.
           - Do not use dummy tautologies to satisfy a property. Forbidden
             theorem conclusions include `True`, `False`, `False = False`,
             `x = x`, and any theorem whose postcondition does not mention the
             modeled function, output state, event list, transfer list, or a
             concrete guard/failure result.

        5. **Theorem Pattern**:
           - Implement theorems inside the `/- [SECTION: THEOREMS] -/` block.
           - Format: `theorem [name] (args...) : [prop] := sorry`
           - End theorem placeholders exactly with `:= sorry`; do not write `:= by sorry`.
           - **Multi-Theorem Policy**: If a single property requires multiple theorems to prove (e.g., helper lemmas), define ALL of them.
           - **STRICTLY FORBIDDEN**: Do NOT define a theorem as `True`, `False`,
             `False = False`, `x = x`, or another placeholder/tautology.
           - Prefer local transition theorems over global reachability
             invariants. Good shapes:
             `fn s ctx args = CallResult.success out → out.state.field = ...`
             `guard_hypotheses → fn s ctx args = CallResult.failure Error.foo`
             `fn s ctx args = CallResult.success out → out.events = [...]`
           - When writing a success-implies postcondition for balance deltas,
             include all disjointness hypotheses needed for map updates. If the
             theorem omits those hypotheses, use a weaker postcondition that is
             true for both equal and distinct addresses.
             Do not write exact sender/from balance decrease theorems for token
             transfer or transferFrom unless the statement also assumes the debit
             address and credit address are distinct.
           - When writing a failure theorem, include all earlier guards as
             hypotheses. Example shape:
             `s.initialized = true → ctx.sender = s.owner →
              s.saleState = SaleState.saleClosed →
              fn s ctx = CallResult.failure Error.notOpen`
           - Only write `ReachC` theorems when the statement includes explicit
             well-formed initial-state assumptions strong enough for the base
             case and every inductive step.
           - Only write aggregate batch-distribution theorems when the theorem
             states the required list invariants (`length` alignment, no duplicate
             recipients/team members, percentage sum/bounds). Otherwise write a
             local theorem about one successful update or the exact fold result.
           - Keep theorem count focused. If a theorem budget is provided, obey it
             strictly. Otherwise generate roughly 2-3 local theorem obligations
             for each selected property, plus helper lemmas only when they are
             directly needed by those obligations.
        """

        user_prompt = f"""
        [INPUT SOLIDITY]
        {solidity_code}

        [PROPERTIES TO VERIFY]
        {props_text}

        [TASK]
        Generate 'Definitions.lean' using the [SECTION] markers.
        1. Put all structs/functions under definitions.
        2. {theorem_task_line}
        3. End every theorem exactly with `:= sorry`, not `:= by sorry`.
        4. Before returning, mentally compile the file under plain Lean 4:
           no forward references, every enum equality has `deriving DecidableEq`,
           no equality derivation for function-valued structures, and no dummy
           tautology theorems.
        """

        print("   LeanFormalizer is architecting the FSM & Theorems (Structured Mode)...")
        return self._query_llm(system_prompt, user_prompt)

# ==========================================
# 3. Lean Prover
# ==========================================
class LeanProver(BaseAgent):
    def __init__(self, model_name: str = None, api_key: str = None, base_url: str = None):
        # Forward the API key and base URL to the base class.
        super().__init__(model_name=model_name, api_key=api_key, base_url=base_url)

    def prove_theorem(self, background_defs: str, target_theorem: str, error_feedback: str = "") -> str:
        feedback_section = ""
        if error_feedback:
            feedback_section = f"""
            [PREVIOUS FAILED ATTEMPT]
            {error_feedback}
            """

        system_prompt = """
        You are a Lean 4 Expert Prover specialized in Smart Contract Verification.

        [TASK]
        Prove the target theorem provided by the user. The theorem typically verifies a specific transition of a Functional State Machine.

        [THE GOLDEN STRATEGY FOR STATE MACHINES]
        Follow this standardized proof flow:

        1. **Unfold & simplify monads**:
           Start with `dsimp [function_name]`.
           If the code uses `do` notation or `Except`, add `simp [Bind.bind, pure, Except.bind, Except.pure]`.

        2. **Prune Branches (The Guard Pattern)**:
           Instead of splitting every `if`, use hypotheses to eliminate impossible branches immediately.
           *Example*: If you have `h : amount > 0` and code has `if amount == 0 then ...`, use `simp [Nat.ne_of_gt h]`.

        3. **Structured Branching**:
           Use `by_cases h : condition` for major logic forks (e.g., `by_cases h_owner : ctx.sender = s.owner`).
           Do NOT use `split_ifs` or `split at h`: these are often unavailable or
           brittle in this Lean environment.
           `by_cases` gives you explicit hypotheses to use with `simp`.

        4. **Final Matching**:
           When the function reduces to a result, proving equality is often just `simp` or `rfl`.
           If a hypothesis has the form
           `h : CallResult.success {...} = CallResult.success out`,
           prefer `cases h` or `subst out` after `simp [...] at h`.
           Avoid `injection h with ...` for `CallResult.success` unless you are
           certain the equality is already between constructor applications; it
           frequently fails after unfolding `match`/`if` code.

        [LEAN 4 STANDARD-LIB PITFALLS]
        The environment is plain Lean 4, not a Mathlib-heavy environment.
        Avoid tactics that may be unavailable: `split_ifs`, `omega`, `linarith`,
        `aesop`, `tauto`, `simpa?`, `native_decide`, `by_contra`.
        Also avoid anonymous `split`/`split at h`; use named `by_cases` instead.
        Prefer robust core tactics: `intro`, `intros`, `cases`, `rcases`,
        `constructor`, `exact`, `apply`, `rw`, `simp`, `dsimp`, `by_cases`,
        `subst`, `rfl`, `exfalso`.

        [INDUCTIVE CASE SPLITS]
        When case-splitting an input inductive, always name constructor fields.
        Do not write anonymous bullets that later refer to missing variables.
        Use this style:

        ```
        cases input with
        | startDistribution => ...
        | finishDistribution => ...
        | getTokens => ...
        | receiveEth => ...
        | transfer to amount => ...
        | withdraw => ...
        ```

        If constructor names differ, inspect the `[BACKGROUND DEFINITIONS]` and
        bind every constructor argument explicitly in its own branch.

        [RECURSIVE REACHABILITY PROOFS]
        For an inductive reachability predicate, first prove a local one-step
        preservation lemma inside the proof when possible:

        ```
        have step_preserves :
            ∀ {s ctx input out}, transition s ctx input = CallResult.success out →
              out.state.someField = s.someField := by
          intro s ctx input out h
          cases input with
          | transfer to amount =>
              dsimp [transition, transferTokens] at h
              by_cases hAddr : to = zeroAddress <;> simp [hAddr] at h
              ...
          | _ =>
              dsimp [transition, ...] at h
        ```

        Then use induction on the reachability hypothesis. Do not refer to
        implicit constructor variables such as `input` unless they are explicitly
        bound in the case pattern. If Lean rejects a constructor pattern, simplify
        it to `| refl => ...` and `| step hReach hStep hNext ih => ...`, then use
        `rw [hNext]` plus the one-step lemma.

        [BALANCE UPDATE PROOFS]
        For map-like balances encoded as functions, unfold the setter and use the
        account inequality hypotheses:
        `simp [setBalance, h_ne, Ne.symm h_ne]`.
        For self-transfer-sensitive statements, make sure a distinct-account
        hypothesis is used; otherwise the theorem may double-count one account.

        [SUCCESS-SHAPE PROOFS]
        For a theorem with `h : transition ... = CallResult.success out`, first
        reduce `h` to the exact success branch, then extract the output and only
        then prove field facts. A robust pattern is:

        ```
        intro h
        have h' := h
        dsimp [transition, helper] at h'
        simp [guardHyp1, guardHyp2] at h'
        cases h'
        simp [getter, setter, Ne.symm h_ne]
        ```

        Do not use unbound `out`, `?out`, record field placeholders, or
        `nomatch` tricks. If the target is an existential success-shape theorem,
        build the concrete `out` with `refine ⟨..., ?_⟩`; Lean tactic
        placeholders are allowed only when every subgoal is solved in the same
        proof.

        [PARSER AND CORE-LIB PITFALLS]
        - Avoid pipeline field access in proof scripts. Do not write
          `expr |>.field`; bind the intermediate value with `let u := expr` or
          use ordinary parenthesized field access if the background definitions
          already use that shape.
        - Avoid the tactic `set ... with ...` and avoid depending on names it
          creates. Use `let localName := expression` in term context, or repeat
          the expression in `simp`.
        - Avoid unavailable or version-sensitive helper names such as
          `Bool.eq_false_of_ne_true`, `Nat.gt_of_lt`, `Nat.ge_of_le`,
          `le_of_not_gt`, bare `not_lt_of_ge`, or Mathlib-only
          arithmetic lemmas. For Bool guards, split with `cases h : b` or
          `by_cases hb : b = true`; for Nat inequalities prefer core names like
          `Nat.not_lt_of_ge`, `Nat.not_lt_of_le`, and direct `simp` from the
          given hypothesis.
        - If the target involves a recursive list helper, unfold the exact helper
          and prove the branch by pattern matching on the list and constructor
          fields. Do not introduce a new recursive helper with a name that may
          conflict with existing definitions.

        [REPAIR MODE WHEN LEAN REJECTS A PREVIOUS ATTEMPT]
        When compiler feedback is provided, fix the exact Lean failure instead of
        restating the same proof.
        - If Lean says `no goals to be solved`, your previous script likely solved
          the theorem before its final tactic. Return the shorter prefix, often
          just `by rfl`, `by simp`, or the proof up to the tactic that closed the
          goal. In impossible branches, prefer `exfalso` before simplifying a
          contradictory success hypothesis, and do not add `cases h` after
          `simp ... at h` if Lean already closed the branch.
        - If Lean says `simp made no progress`, do not repeat the same `simp`.
          Either unfold the specific definitions mentioned in the theorem, use a
          named `by_cases` for the branch guard, or replace the final line with
          `rfl`, `simp_all`, or a constructor proof as appropriate.
        - If Lean reports unsolved goals after nested state updates, explicitly
          unfold the map/list setter helpers used in the postcondition, then use
          the equality/inequality hypotheses with `simp [helper, h, Ne.symm h]`.
        - If `rfl` fails after a success equality hypothesis, first simplify the
          hypothesis until it is an equality between constructor applications,
          then use `cases h` or `subst out`, and only then finish with `simp`.
        - For boolean-return external-call guards such as
          `returnsBool && !returnedBool`, if the theorem assumes a disjunction
          like `returnsBool = false \\/ returnedBool = true`, split that
          disjunction and simplify the success hypothesis inside each branch.
          Avoid proving an intermediate boolean equality and then doing
          dependent elimination outside the branch; that often triggers
          `dependent elimination failed`.

        [STRICT OUTPUT FORMAT]
        1. **Code Only**: Output ONLY the Lean code block starting with `by`.
        2. **No Text**: NO conversational text, NO explanations.
        3. **No Headers**: NO `**Proof**`.
        4. Do not introduce dummy proof branches or tautological helpers such as
           `∨ True`, `True ∨`, `have : True := ...`, or `have := rfl`.

        [ONE-SHOT DEMONSTRATION - MIMIC THIS STYLE]

        <Example_Input>
          [BACKGROUND DEFINITIONS]
          structure State where balance : Nat
          def withdraw (s : State) (amount : Nat) : Except String State :=
            if amount > s.balance then throw "Insufficient"
            else Except.ok { s with balance := s.balance - amount }

          [TARGET THEOREM]
          theorem withdraw_success (s : State) (amount : Nat) (h : s.balance >= amount) :
            withdraw s amount = Except.ok { s with balance := s.balance - amount } := sorry
        </Example_Input>

        <Example_Output>
        ```lean
        by
          -- 1. Unfold logic
          dsimp [withdraw]
          -- 2. Prune 'if' using hypothesis (negate the failure condition)
          have h_sufficient : ¬ (amount > s.balance) := Nat.not_lt_of_le h
          -- 3. Simplify using the pruning fact
          simp [h_sufficient]
        ```
        </Example_Output>

        [END OF EXAMPLE]

        Now, prove the user's theorem using this strategy.
        """

        user_prompt = f"""
        [BACKGROUND DEFINITIONS]
        {background_defs}

        [TARGET THEOREM]
        {target_theorem}

        {feedback_section}

        [TASK]
        Replace `:= sorry` with a valid proof (starting with `by`) for the theorem above.
        REMEMBER: Output ONLY the code block.
        """

        if error_feedback:
            print("   LeanProver is repairing the proof...")
        else:
            print("   LeanProver is proving the selected target...")

        # Query through the base class, which removes <think> blocks via _clean_code.
        raw_response = self._query_llm(system_prompt, user_prompt)

        # Apply the original extraction logic as a compatibility safeguard.
        if hasattr(self, '_extract_proof'):
            return self._extract_proof(raw_response)

        # Use a simple fallback extraction when needed.
        if "```" in raw_response:
            blocks = re.findall(r"```(?:lean4?|)\s*(.*?)```", raw_response, re.DOTALL)
            if blocks:
                return blocks[-1].strip()

        return raw_response.strip()
# ==========================================
# 4. Other Agents
# ==========================================
class SolidityCoder(BaseAgent):
    def __init__(self, model_name=DEFAULT_MODEL_NAME, api_key=None, base_url=None):
        super().__init__(model_name=model_name, api_key=api_key, base_url=base_url)

    def generate_code(
        self,
        requirements: str,
        constraints: Optional[List[str]] = None,
    ) -> str:
        constraints = constraints or []
        constraint_text = "\n".join([f"- {c}" for c in constraints]) if constraints else "None"

        # Emphasize preservation of the original structure in the system prompt.
        system_prompt = """
        You are a Solidity Security Engineer.

        [TASK]
        Your goal is to fix vulnerabilities in the provided code while preserving the original intent and structure.

        [CRITICAL GUIDELINES]
        1. **MINIMAL CHANGES**: Only modify the lines necessary to fix the security constraints.
        2. **PRESERVE ARCHITECTURE**: Do NOT change inheritance (e.g., do NOT switch `Ownable` to `Ownable2Step`) unless explicitly asked.
        3. **COMPLIANCE**: Ensure the code remains compatible with the [ORIGINAL REQUIREMENTS].
        4. **TURN FEEDBACK INTO CODE**: Attack-derived properties are mandatory repair requirements, not optional comments.
           Replay every provided counterexample abstractly. For each successful state-changing step, identify:
           - the caller role and whether the caller should be authorized;
           - assets or claims created, moved, burned, reserved, or released;
           - persistent liabilities created or reduced;
           - phase/state-machine changes and whether the transition is allowed;
           - whether the operation is repeatable and whether repeating it preserves invariants.
        5. **GENERIC REPAIR RULES**:
           - Sensitive entry points that change authority, phase, configuration, custody, claims, or asset movement must have an explicit authorization rule derived from the requirement.
           - Accounting repairs must preserve conservation: every successful inflow/outflow or claim update must be reflected exactly once in storage.
           - Settlement/finalization/distribution-style operations must be idempotent or explicitly monotonic; a repeated successful call must not duplicate effects unless the requirement says so.
           - If a requirement says a sale, whitelist, presale, claim, or mint is limited/capped but does not give an exact numeric cap, do not invent an unmarked magic number as if it came from the requirement. Encode explicit per-account accounting and choose a named configurable parameter or named constant with a conservative default, so tests and reviews can inspect the inferred policy separately from the requirement.
           - Arbitrary asset-transfer helper functions should be removed from the public surface, made internal/private, or constrained to an accounting-aware authorized flow.
           - Reverts must not partially update state. Use checks-effects-interactions when external value transfer is involved.
        6. **POST-REPAIR SELF-CHECK**:
           Before returning code, replay the provided failing trace and its minimal generalization.
           The repaired code must either block the invalid transition or make the state/accounting change conserve the stated invariant.
        7. **OUTPUT**: Return ONLY the full corrected Solidity code.
        """

        user_prompt = f"{requirements}\n\n[SECURITY CONSTRAINTS TO FIX]\n{constraint_text}\n"

        print("   SolidityCoder is fixing code...")
        return self._query_llm(system_prompt, user_prompt)
    def generate_modern_code(self, legacy_prompt: str) -> str:
        """
        [Modern-First Strategy]
        Read prompts with legacy requirements while generating Solidity 0.8.20 code.
        """
        system_prompt = """
        You are an Expert Solidity Architect specialized in modernizing legacy code requirements.

        [CRITICAL: OUTPUT SCOPE]
        1. **TARGET ONLY**: You must ONLY implement the TARGET CONTRACT logic (e.g., the Token, the Auction).
        2. **NO TESTS**: DO NOT generate `Test`, `Handler`, or `Invariant` contracts. Those are handled by a separate agent.
        3. **SINGLE FILE**: Output all logic (including helper interfaces/libraries) in ONE code block.
        4. **NO LOCAL IMPORTS**: Do NOT use `import "./OtherFile.sol"`. Define interfaces locally if needed or use OpenZeppelin.
        5. **SPDX**: Ensure there is exactly ONE `// SPDX-License-Identifier: MIT` at the very top.
        6. Unique Naming: Do NOT name events and functions identically.

        [MODERNIZATION RULES]
        1. **Version**: Use `pragma solidity ^0.8.20;`.
        2. **Arithmetic**: REMOVE `SafeMath` library usage. Use Solidity 0.8's built-in overflow protection.
        3. **Syntax**: Use `constructor()`, `emit Event()`, and explicit function visibility.
        4. **Interfaces**: If the prompt mentions interfaces (e.g., ERC20Basic), **DEFINE THEM** in your output code. Do not import them from missing files.
        5. **Explicit Casting**:
           - Never cast `0` directly to a contract type. Use `ContractType(address(0))`.
           - Fix `selfdestruct` usage if possible, or keep it but handle warnings.
        6. **Requirement-Derived Limits**:
           - If the prompt explicitly gives a per-user/per-wallet/presale/whitelist cap, implement that exact cap with per-account accounting.
           - If the prompt only says "limited", "capped", "presale", or "whitelist" without a number, implement per-account accounting and expose the policy as a named configurable parameter or named constant with a conservative default. Do not hide the inferred value in an unexplained magic number.
           - Safety/fuzz agents will test the semantic rule "repeated successful calls by the same account cannot exceed the implemented cap"; they must not assume a specific numeric cap unless the requirement provides it.
        [OUTPUT]
        Return ONLY the complete Solidity code block for the Target Contract. No Markdown explanations.
        """

        user_prompt = f"""
        [LEGACY REQUIREMENT]
        {legacy_prompt}

        [TASK]
        Rewrite the business logic above using Solidity 0.8.20.
        DO NOT include any test code.
        """

        print("   SolidityCoder is modernizing the legacy requirement to 0.8.20...")
        return self._query_llm(system_prompt, user_prompt)

class AuditorAgent(BaseAgent):
    def audit_unproven(self, code: str, theorems: List[str]) -> str:
        return self._query_llm("You are an Auditor.", f"Analyze failed theorems: {theorems}\nCode: {code}")

    def audit_contract(self, code: str, properties: List[str]) -> Dict:
        system_prompt = """
        You are a senior smart contract security auditor.

        [TASK]
        Audit the Solidity contract for high-severity vulnerabilities only.
        Focus on exploitable fund loss, privilege escalation, permanent lockup, insolvency,
        reentrancy, unchecked external calls, broken authorization, and severe accounting errors.

        [OUTPUT FORMAT]
        Return ONLY valid JSON:
        {
          "high_severity_count": 0,
          "medium_severity_count": 0,
          "low_severity_count": 0,
          "findings": [
            {"severity": "High", "title": "...", "evidence": "..."}
          ],
          "audit_pass": true
        }
        """
        user_prompt = f"""
        [SELECTED PROPERTIES]
        {json.dumps(properties, indent=2)}

        [SOLIDITY CODE]
        {code}
        """
        response = self._query_llm(system_prompt, user_prompt)
        try:
            data = json.loads(response.replace("```json", "").replace("```", "").strip())
            data["audit_pass"] = int(data.get("high_severity_count", 0)) == 0
            return data
        except Exception as e:
            return {
                "audit_pass": None,
                "high_severity_count": None,
                "medium_severity_count": None,
                "low_severity_count": None,
                "findings": [],
                "error": f"Failed to parse audit JSON: {e}",
                "raw": response,
            }

# ==========================================
# 5. Environment Architect
# ==========================================
class EnvironmentArchitect(BaseAgent):
    """Generate simulation artifacts from the target contract interface."""

    def __init__(self, model_name=None, api_key=None, base_url=None):
        super().__init__(model_name, api_key, base_url)

    def extract_interface_description(self, solidity_code: str) -> str:
        """Extract a human-readable interface summary for simulation agents."""
        system_prompt = """
        You are a Solidity Interface Analyzer.
        Your job is to extract a concise Interface Summary from the raw source code.

        [TASK]
        1. List all `public` or `external` functions with their full signatures (params and types).
        2. List all `public` state variables (they have implicit getters).
        3. List all `events`.
        4. Identify the `constructor` arguments.

        [OUTPUT FORMAT]
        Contract Name: [Name]
        Functions:
        - functionName(type param1, type param2) -> (returnType)
        ...
        State Variables:
        - variableName (type)
        ...
        Constructor Args:
        - (type arg1, type arg2)
        """
        user_prompt = f"[TARGET CONTRACT]\n{solidity_code}"

        print("   Architect is extracting contract interface...")
        return self._query_llm(system_prompt, user_prompt)

    def generate_mocks(self, solidity_code: str, error_feedback: str = "") -> str:
        system_prompt = """
        You are a Solidity Mock Engineer.
        Generate a 'Mocks.sol' file to provide concrete implementations for dependencies found in the Target Contract.

        [CRITICAL: NAMESPACE ISOLATION]
        The Target Contract (`Target.sol`) and this file (`Mocks.sol`) will be imported into the same Script.
        To avoid "Identifier already declared" compiler errors, you MUST follow these rules:

        1. **NO REDEFINITIONS**:
           - **DO NOT** define `interface IERC20`, `interface IERC721`, `contract Ownable`, etc., if they are likely present in the Target.
           - Instead, define **Unique Interfaces** for your mocks.

        2. **NAMING CONVENTION**:
           - Use `IMock` prefix for interfaces.
           - Use `Mock` prefix for contracts.
           - Example:
             ```solidity
             // INSTEAD OF: interface IERC20 { ... }
             interface IMockERC20 {
                 function balanceOf(address) external view returns (uint256);
                 function transfer(address, uint256) external returns (bool);
                 // ... other methods
             }

             contract MockERC20 is IMockERC20 { ... }
             ```

        [CRITICAL: MOCK IMPLEMENTATION DETAILS]
        1. **View vs Pure**: Ensure `factory()` and `WETH()` are `view` functions, NOT `pure`, to allow state usage if needed.
        2. **State Variables**: If an interface defines a function (e.g., `WETH()`), implement it as a function returning a constant or state variable, do not define it as a public variable if the interface expects a function.
        3. **Valid Reverts**: Solidity has `revert()` and `revert(string)`, but no `revert(bytes)` overload.
           - NEVER emit code like `revert(bytes(reason))`.
           - If a mock needs a configurable revert reason, store it as `string public revertMessage;` and call `revert(revertMessage);`.
           - If raw bytes are truly required, use a custom error or inline assembly; prefer string reasons for mocks.

        [TASK]
        1. Analyze the [TARGET CONTRACT] to see what external calls it makes (e.g., Uniswap Pair, ERC20 token, Oracle).
        2. Generate `Mocks.sol` containing `MockERC20`, `MockPair`, etc.
        3. Ensure `Mocks.sol` is self-contained (defines its own `IMock...` interfaces) and does not import `Target.sol`.
        4. If NO external dependencies are found, output: `// No mocks required`.
        """

        feedback = f"\n\n[PREVIOUS COMPILER ERROR]\n{error_feedback}" if error_feedback else ""
        user_prompt = f"[TARGET CONTRACT]\n{solidity_code}{feedback}\n\n[OUTPUT] Mocks.sol code block."
        print("   Architect is designing Mocks...")
        return self._normalize_erc721_mock_approval(
            self._normalize_solidity_code(self._query_llm(system_prompt, user_prompt))
        )

    def generate_safety_rules(self, solidity_code: str, properties: List[str], error_feedback: str = "") -> Tuple[str, str]:
        # 1. Prepare the property text list.
        props_text = "\n".join([f"- {p}" for p in properties])

        # 2. Build the address-definition block dynamically.
        agents = get_simulation_agents()
        address_definitions = []
        array_assignments = []

        for idx, (name, data) in enumerate(agents.items()):
            # Generate: address alice = 0x123...;
            address_definitions.append(f"          address {name.lower()} = {data['address']}; // {data['role']}")
            # Generate: users[0] = alice;
            array_assignments.append(f"          users[{idx}] = {name.lower()};")

        # Convert the generated lines to a single string.
        address_block_str = "\n".join(address_definitions)
        array_block_str = "\n".join(array_assignments)
        agent_count = len(agents)
        # Add the repair-feedback context.
        feedback_section = ""
        if error_feedback:
            feedback_section = f"""
            [PREVIOUS COMPILATION ERROR]
            Your previous SafetyRules code failed to compile.
            ERROR LOG:
            {error_feedback}

            [INSTRUCTION]
            FIX the code based on the error above. Ensure imports are correct ("./Target.sol") and syntax is valid.
            """
        # 3. Build the prompt with instructions for invariant templates.
        system_prompt = f"""
        You are a Solidity Security Engineer.
        You need to create a 'Referee' system for a simulation.

        {feedback_section}

        [FILE SYSTEM REQUIREMENT - CRITICAL]
        The target contract is ALWAYS saved as "Target.sol".
        You MUST use this exact import in SafetyRules.sol:
        import "./Target.sol";
        Do not import Mocks.sol unless SafetyRules directly references a mock type.

        [SOLIDITY SYNTAX RULES - CRITICAL]
        - If the concrete target contract declares `receive() external payable` or `fallback() external payable`,
          cast through a payable address when storing a concrete target reference:
          `target = ConcreteTarget(payable(target_));` or `target = ConcreteTarget(payable(address(target_)));`.
          Interface casts can remain non-payable.
        - Never write declarations like `Type /*name*/ = expr;`. If you need a local variable, write
          `Type name = expr;` and optionally add `name;` on the next line to silence warnings.
        - Never emit an empty named import such as `import {{ }} from "./X.sol";` or
          `import {{ /* none */ }} from "./X.sol";`. Omit it or use `import "./X.sol";`.

        [TASK 1: SafetyRules.sol]
        Write a contract `SafetyRules` that holds references to the Target and Mocks.

        [CRITICAL: IMPLEMENTING INVARIANTS]
        You will receive a list of properties to verify. **Most properties include an `{{Invariant Template: ...}}` field.**
        1. **APPLICABILITY FIRST**:
           - Before implementing each property, decide whether it is applicable to THIS target contract and THIS requirement.
           - SafetyRules functions are view-only state referees. If a property needs a caller-specific prank,
             transaction history, event logs, gas measurement, source-code inspection, or mutating execution,
             the check function MUST return `(true, "N/A: ...")` and explain the missing evidence.
           - NEVER encode "missing optional interface/standard feature" as `(false, ...)` unless the target
             explicitly claims that interface/standard or the user requirement requires it.
           - Do NOT fail a role or architecture property merely because a role, phase, or helper exists.
             Fail only when the current public state proves a concrete invariant violation, or when a concrete
             requirement can be checked from exposed state.
        2. **USE THE TEMPLATE ONLY WHEN APPLICABLE**: You MUST implement the logic described in the template when it fits the target.
           - If the template is mathematical (e.g., `sum(balances) <= total_supply`), write a loop in Solidity to sum user balances and compare.
           - If the template is code-based (e.g., `unchecked {{ require(...) }}`), adapt it to verify the target's state.
        3. **ADAPT TO CONTEXT**:
           - Replace abstract terms in the template (like `balance`) with actual calls to the Target (e.g., `target.balanceOf(users[i])`).
           - Use the `users` array (passed as argument) to iterate over all agents if needed.
           - Do NOT use low-level `staticcall` probes for functions the target does not expose just to prove absence. Absence is not a violation by itself.
        4. **NO SELF-FULFILLING FAILURES**:
           - A check function must not simply `return (false, "...")` based on source-code style, architecture preference, or untestable assumptions.
           - Every `(false, reason)` must be tied to an observed on-chain state exposed through public getters
             or address balances, not to a hypothetical call that this view-only referee did not execute.
        5. **ROUTING NON-VIEW PROPERTIES**:
           - For unauthorized-caller, revert-preservation, event-emission, gas-bound, reentrancy-order, and
             idempotency-under-repetition properties, prefer returning N/A here. Those belong in Foundry
             handler tests or attack scripts where calls can be executed and snapshots can be taken.
        6. **SEMANTIC OBLIGATIONS**:
           - Properties prefixed with `[SEMANTIC-...]` are requirement-derived semantic obligations.
             Treat them as useful lower-confidence checks, not as source-code style complaints.
           - Implement a concrete view check only when public getters or address balances expose enough
             state. Good examples are liabilities/assets inequalities, total claimable balances for
             tracked users, per-user minted amount when a getter exists, or nonzero code at exposed
             created/proxy addresses.
           - Do NOT satisfy a semantic obligation with a mirror-only invariant. For example, if the
             obligation is solvency/backing, prefer `sum(claims) <= address(target).balance` or a
             token-backing inequality over merely checking that a ghost counter equals a target counter.
           - If the semantic obligation needs execution history or unavailable private state, return
             `(true, "N/A: semantic obligation requires Foundry execution/history: ...")`.
           - Prefix violation messages for these checks with `SEMANTIC:` so experiment analysis can
             separate inferred semantic checks from hard taxonomy properties.
        7. **FUNCTION SIGNATURE**:
           - Create a view function for EACH property: `function checkInvariant_[ID](address[] memory users) public view returns (bool, string memory)`.
           - Sanitize `[ID]` into a valid Solidity identifier by replacing dots, hyphens, spaces, and brackets with underscores.
           - Return `(true, "")` if safe.
           - Return `(true, "N/A: reason")` if not applicable.
           - Return `(false, "Description of violation")` if unsafe.

        [TASK 2: Check Logic]
        Write a snippet of Solidity code that calls ALL the check functions you just wrote.
        This snippet will be inserted into a Foundry script's `run()` function.
        Assume `rules` is an instance of `SafetyRules`.
        If a check function returns `(true, "N/A: ...")`, the snippet must NOT fail; it may log the note and continue.

        [CRITICAL REQUIREMENT: ARGUMENT PREPARATION]
        If your check functions require arguments (e.g., `address[] memory users`), YOU MUST DEFINE THEM in the snippet.
        - You MUST use the following HARDCODED addresses for the agents (Environment Configuration):

{address_block_str}

        Example Snippet:
           // 1. Prepare Data
           address[] memory users = new address[]({agent_count});
{array_block_str}

           // 2. Run Checks
           (bool success, string memory err) = rules.checkInvariant_L1_1_Overflow(users);
           if (!success) console.log(err);
           require(success, err);

        [OUTPUT FORMAT]
        Return a JSON object:
        {{
           "safety_rules_code": "... full contract code ...",
           "check_script_body": "... logic snippet including variable definitions ..."
        }}
        """

        user_prompt = f"[TARGET]\n{solidity_code}\n\n[PROPERTIES]\n{props_text}"

        print("   Architect is drafting Safety Rules & Check Script using Templates...")
        response = self._query_llm(system_prompt, user_prompt)

        try:
            # Attempt to parse the response as JSON.
            cleaned_json = response.replace("```json", "").replace("```", "").strip()
            data = json.loads(cleaned_json)
            safety_rules_code = self._normalize_solidity_code(data["safety_rules_code"])
            safety_rules_code = self._normalize_payable_target_casts(safety_rules_code, solidity_code)
            check_script_body = self._normalize_solidity_code(data.get("check_script_body", ""))
            return safety_rules_code, check_script_body
        except Exception as e:
            print(f"   Error parsing Safety Rules: {e}")
            return "", "// Parsing failed, skipping checks"

    def _normalize_address_patterns(self, patterns: Dict[str, str]) -> Dict[str, str]:
        normalized: Dict[str, str] = {}
        for label, pattern in (patterns or {}).items():
            if not isinstance(pattern, str):
                continue
            normalized[label] = re.sub(
                r"0x(?P<class>\[[^\]]+\])(?P<count>\d+)",
                r"0x\g<class>{\g<count>}",
                pattern,
            )
        return normalized

    def generate_deploy_script(self, target_code: str, mocks_code: str, safety_code: str, error_feedback: str = "") -> Tuple[str, Dict[str, str]]:
        # 1. Generate account-funding logic dynamically from config.py.
        # Use vm.deal to avoid panics caused by an underfunded script contract.
        agents = get_simulation_agents()
        funding_logic = []
        agent_list_str = []

        for name, data in agents.items():
            addr = data['address']
            # Include actor-role information in the prompt.
            agent_list_str.append(f"- {name} ({data['role']}): {addr}")

            # Generate the Solidity account-funding line.
            # Use vm.deal(address, amount) instead of payable(addr).transfer(amount).
            funding_logic.append(f"        vm.deal({addr}, 100 ether); // Fund {name}")

        funding_block = "\n".join(funding_logic)
        agent_context_block = "\n".join(agent_list_str)

        # 2. Incorporate error feedback.
        feedback_section = ""
        if error_feedback:
            feedback_section = f"""
            [PREVIOUS COMPILATION ERRORS]
            The previous Deploy.s.sol script failed to compile.
            Fix these specific errors:
            {error_feedback}
            """

        # 3. Build the prompt.
        system_prompt = f"""
        You are a Foundry Deployment Expert.
        Write a `Deploy.s.sol` script to deploy Target, Mocks, and SafetyRules.

        {feedback_section}

        [FILE SYSTEM LAYOUT - CRITICAL]
        You MUST use these exact import paths:
        1. Target Contract: import {{ [ContractName] }} from "../src/Target.sol";
        2. Mocks: import {{ ... }} from "../src/Mocks.sol";
        3. SafetyRules: import {{ SafetyRules }} from "../src/SafetyRules.sol";
        If no mock symbols are required, do not write `import {{ }} from "../src/Mocks.sol";` or
        `import {{ /* no mocks */ }} from "../src/Mocks.sol";`. Omit the import or use `import "../src/Mocks.sol";`.

        [TASK]
        1. Analyze `Target` constructor to see what Mocks it needs.
        2. Deploy Mocks first.
        3. Deploy Target (injecting Mock addresses).
        4. Deploy SafetyRules (injecting Target/Mock addresses).

        [AGENTS CONTEXT]
        The following agents will participate in the simulation:
{agent_context_block}

        [CRITICAL RULE: FUNDING ACCOUNTS]
        - To fund test accounts (Alice, Bob, etc.), you **MUST** use `vm.deal(address, amount)`.
        - **DO NOT** use `payable(address).transfer(amount)`.
        - REASON: The deployment script contract itself has 0 ETH, so calling `.transfer()` will fail with `<empty revert data>`.
        - Only `vm.deal` works reliably in simulation scripts.

        [CRITICAL RULE: BROADCAST SIGNER]
        - The deployment command already passes `--private-key`.
        - Use `vm.startBroadcast();` or a deployer private key, never `vm.startBroadcast(alice)`, `vm.startBroadcast(bob)`, or any agent address.
        - Agent addresses are scenario actors, not unlocked deployer wallets.

        [CRITICAL REQUIREMENT: FUND THE AGENTS]
        In your `run()` function, inside the broadcast block, YOU MUST INCLUDE THIS EXACT FUNDING LOGIC:

{funding_block}

        [LOGGING REQUIREMENT]
        You must log the deployed addresses using `console.log("TAG:", address)`.
        The TAGs must be unique (e.g., TARGET, TOKEN_1, SAFETY_RULES).

        [OUTPUT FORMAT]
        Return a JSON object:
        {{
           "script_code": "... solidity code ...",
           "address_patterns": {{
                "TARGET": "TARGET:\\s+(0x[a-fA-F0-9]{40})",
                "SAFETY_RULES": "SAFETY_RULES:\\s+(0x[a-fA-F0-9]{40})",
               ...
           }}
        }}
        """

        user_prompt = f"""
        [TARGET CODE]\n{target_code}
        [MOCKS CODE]\n{mocks_code}
        [SAFETY CODE]\n{safety_code}
        """

        print("   Architect is configuring Deployment...")
        response = self._query_llm(system_prompt, user_prompt)

        try:
            cleaned_json = response.replace("```json", "").replace("```", "").strip()
            data = json.loads(cleaned_json)
            script_code = self._normalize_solidity_code(data["script_code"])
            print(f"   [DEPLOY SCRIPT]:\n{script_code[:300]}...\n")
            return script_code, self._normalize_address_patterns(data.get("address_patterns", {}))
        except Exception as e:
            print(f"   Error parsing Deployment config: {e}")
            return "", {}

    # EnvironmentArchitect helper for interface extraction.
    def generate_fuzz_test(self, target_code: str, safety_code: str, error_feedback: str = "", previous_code: str = "", semantic_context: str = "") -> str:
        feedback_section = ""
        if error_feedback:
            feedback_section = f"""
            [PREVIOUS COMPILATION ERROR]
            Your previous Invariant Test failed to compile.
            ERROR LOG:
            {error_feedback}

            [PREVIOUS INVARIANT.T.SOL]
            {previous_code if previous_code else "(not provided)"}

            [INSTRUCTION]
            FIX the harness based on the error above.
            - Output a COMPLETE replacement `Invariant.t.sol`, not a patch or fragment.
            - Do not change the target contract semantics; repair only the Foundry harness.
            - Preserve valid ghost-modeling decisions from the previous version where possible.
                - Ensure `targetSelector` uses `bytes4[]` dynamic array correctly.
                - Ensure imports are from `./Target.sol`.
                - Ensure all functions, loops, declarations, and contracts are syntactically complete.
                - Do not import `StdInvariant`, `Console`, or `forge-std/Console.sol`.
                - `InvariantTest` must inherit only `Test`; do not write `is Test, StdInvariant` or `is StdInvariant`.
                - Do not declare a public array and a manual getter with the same name.
                """

        system_prompt = f"""
        You are an advanced Foundry Invariant Test Architect.
        Your goal is to generate a **Generic Handler-Based Invariant Test Suite** for *ANY* given Solidity contract.

        {feedback_section}

        [ARCHITECTURAL PATTERN: THE HANDLER]
        You must implement two contracts in one file: `Handler` and `InvariantTest`.

        1. **Analyze the Target Contract**:
           - Identify state-changing functions (public/external).
           - Identify core accounting variables (e.g., `balances`, `totalSupply`).

        2. **Implement `contract Handler is Test`**:
           - **Ghost Variables**: Create shadow variables (e.g., `mapping(address => uint) public ghost_balance`) to track expected state.
           - **Function Wrappers**:
             - Create wrappers for EACH target function to act as entry points for the fuzzer.
             - **Pattern**:
               1. Randomly select valid parameters (using `bound()`, `seed % actors.length`).
               2. Snapshot every ghost variable and tracked collection that might change.
               3. Call Target with `try/catch`.
               4. **Commit-on-success rule**: Prefer updating ghost variables ONLY inside the `try` success branch, after the target call returned.
               5. If you must compute expected deltas before the call, store them in local variables only; do not write ghost state until success.
               6. On `catch`, leave ghost state unchanged or restore the full snapshot. Reverted calls must not produce ghost deltas.

            3. **Implement `contract InvariantTest is Test`**:
               - `setUp()`: Deploy Target and Handler.
               - **Fuzzer Configuration (CRITICAL)**: Use `targetSelector` to restrict calls *only* to the Handler.
               - **Invariants**: Write checks like `invariant_solvency()` comparing `target.var()` vs `handler.ghost_var()`.

        [CRITICAL IMPLEMENTATION DETAILS - DO NOT IGNORE]
        1. **File Imports**:
           - The target contract is ALWAYS located at `"./Target.sol"`.
               - **CORRECT**: `import {{ContractName}} from "./Target.sol";`
               - **WRONG**: `import "./BasicToken.sol";` (This file does not exist).
               - **CORRECT Foundry import**: `import "forge-std/Test.sol";`
               - **WRONG**: `import {{StdInvariant}} from "forge-std/Test.sol";`
               - **WRONG**: `import {{Console}} from "forge-std/Console.sol";`
               - Do not import or inherit `StdInvariant`; `Test` provides the needed invariant helpers in this project.

        2. **Selector Initialization (Fixing Compilation Errors)**:
           - Solidity strictly distinguishes `bytes4[1]` (fixed) and `bytes4[]` (dynamic). `FuzzSelector` requires a DYNAMIC array.
           - The dynamic array length must exactly equal the number of assigned selectors. Do not over-allocate the array; any unassigned slot becomes `0x00000000` and Foundry will fail before running invariants.
           - Never include `bytes4(0)` or `0x00000000` in the selector list unless you intentionally implement and test a fallback dispatcher.
           - **WRONG**: `selectors: [handler.action.selector]` (Causes implicit conversion error).
           - **CORRECT**:
             ```solidity
                   bytes4[] memory selectors = new bytes4[](1);
                   selectors[0] = handler.action.selector;
                   targetSelector(FuzzSelector({{addr: address(handler), selectors: selectors}}));
                 ```

        3. **Ghost State Sync (ANTI-FALSE-POSITIVE RULES)**:
           - Always assume the Target might revert.
           - DO NOT optimistically mutate ghost mappings/arrays/counters before the call unless you snapshot and restore every affected key/array length/counter.
           - The safest pattern is:
             ```solidity
             uint256 delta = ...;
             try target.action(args) {{
                 ghost_counter += delta;
                 ghost_balance[user] += delta;
             }} catch {{
                 // no ghost mutation
             }}
             ```
           - For arrays, record `uint256 oldLen = arr.length`; if reverting after an optimistic push, pop until `arr.length == oldLen`.
              - For duplicate recipients/users in one call, aggregate deltas per unique address before comparing, or update ghost state in the exact same order as the target only after success.
              - Do NOT snapshot per-recipient balances in an array indexed by call position if the same address may appear twice. That stale-snapshot pattern overwrites the first delta on duplicate recipients and creates false positives. Either choose unique recipients or use `ghost_balance[to] += qty` after the target call succeeds.
           - For transfers, aliasing is part of the EVM semantics. If `from == to`,
             the ghost balance for that address must match the target's actual
             self-transfer behavior; do not subtract and then add using two stale
             pre-call snapshots for the same key. If exact self-transfer semantics
             are not modeled, exclude self-transfers from that handler action with
             a clear guard.
           - If multiple actor labels, recipients, owners, or tracked users may
             resolve to the same address, track and compare ghost state by unique
             address, not by label or array index.
           - If the Handler keeps a deduplicated actor/holder array and inserts
             `owner` into that array, do NOT also add `owner` separately in
             balance-sum invariants. Every independent balance sum must count
             each address at most once.
           - Do NOT declare a local variable whose type is `mapping(...)` inside a
             constructor or function. Solidity mappings can only be state variables
             or references to existing storage mappings; they cannot be created
             dynamically. For uniqueness in setup, use a private array plus a
             linear-scan `_pushUnique(address)` helper, or a state mapping.
           - Do not use unbounded `while` loops for "unique random selection".
             Modular strides such as `(seed % actors.length) + 1` can fail to
             visit enough unique actors and hang the fuzz campaign. Prefer a
             bounded `for` loop over the actor array, or use `stride = 1` with
             `count <= actors.length`.
           - Never create an invariant comparing target state to a ghost value for users that the handler does not fully track.
           - If the Handler uses `vm.warp`, time must be monotonic forward only. Never warp to an arbitrary absolute timestamp that can be lower than the current timestamp. Use a bounded delta or `if (next > block.timestamp) vm.warp(next);`.
           - Do not assert that a lazy `currentState` getter must always equal the timestamp-derived state after arbitrary fuzz warps. Many contracts update lifecycle state only when a mutating function runs. Only assert state/time consistency immediately after a transition action you performed, or use conservative one-way checks that cannot be invalidated by stale state.
           - Treat owner/admin configuration setters for caps, limits, thresholds, prices, rates, and max values as mutable policy changes, not as retroactive constraints on past actions. If your handler includes a setter that can lower a cap/limit after users already minted/claimed/deposited, do NOT write an invariant like `historicalUserCount <= currentCap`; that creates a false positive. Either omit that setter from the fuzz selector, make the wrapper keep the value non-decreasing relative to already-observed usage, or check the cap only as a wrapper-local pre/post property for the mint/buy/claim call that uses the current cap.

            4. **Loop Bounds (Fixing Type Errors)**:
               - When iterating over actors/users, ensure you compare the loop counter (`uint`) with a LENGTH (`uint`).
               - **WRONG**: `i < handler.actors(0)` (Comparing uint vs address -> Compiler Error).
               - **CORRECT**: `i < handler.actorsLength()` or expose a public count variable.
               - If you declare `address[] public actors;`, Solidity already creates `actors(uint256)`.
                 Do NOT also define `function actors(uint256)` manually. If you need a manual getter,
                 make the array private and name the getter `actorAt(uint256)`.

            5. **You MUST use targetContract(address(handler)) in setUp() to ensure the Fuzzer ONLY interacts with the Handler. Do NOT allow direct calls to the Target, otherwise ghost variables will get out of sync."

            6. **Foundry Base Contract Rules**:
               - `Handler` may inherit `Test` if it needs `vm`, `bound`, or `makeAddr`.
               - `InvariantTest` MUST inherit exactly `Test`.
               - Never write `contract InvariantTest is StdInvariant`; that loses `vm`, `makeAddr`, `assertEq`, and `assertLe`.
               - Never write `contract InvariantTest is Test, StdInvariant`; it can cause inheritance linearization errors.

        [CRITICAL: HANDLER ACTOR CONTEXT] When writing Handler functions with the useActor modifier:
        NEVER use msg.sender inside the handler function. vm.startPrank does NOT change msg.sender of the current execution frame.
        Instead, define the actor as a local variable or pass it from the modifier.
        In `setUp()`, create explicit actors and fund every actor that may call payable functions with `vm.deal(actor, 100 ether)`;
        positive payable paths must be reachable before invariants judge security.

        [CRITICAL: PAYABLE TARGET CASTS]
        - If the target contract declares `receive() external payable` or `fallback() external payable`, Solidity requires concrete casts through payable addresses.
        - Use `ConcreteTarget(payable(address(target)))` or `ConcreteTarget(payable(targetAddress))` when casting to that concrete type.
        - Do not write `ConcreteTarget(address(target))` for a contract with payable receive/fallback; Foundry will reject it before running invariants.
        - Do not attach `{{value: 0}}` to nonpayable calls. A zero-value call option still requires the function to be payable; use plain `target.fn(args)` when no ETH is being sent.
        - For payable target calls, if a real price getter returns 0, that is a valid exact price. Do not replace a successful zero price with a probe payment.

        [PROPERTY APPLICABILITY]
        - Invariants must check behavior that is actually exposed by the target's public interface.
        - Do NOT assert ERC20/ERC721 standard behavior unless the target explicitly implements those interfaces or exposes the standard functions.
               - Do NOT assert accounting identities that your Handler cannot model exactly. If rounding, duplicates, or partial distribution are hard to model, either model them exactly or omit that invariant.
               - Do not include two wrappers for the same target function in the fuzz selector. Pick one conservative wrapper per target function; duplicate wrappers with different ghost semantics make root-cause analysis unreliable.

        [SEMANTIC OBLIGATION HANDLING]
        - If [SEMANTIC OBLIGATIONS] are provided, use them as scenario objectives for the Handler.
        - Add goal-directed wrappers for valid positive paths before random adversarial paths: owner/admin setup, opening sale/distribution phases, whitelisting/allowlisting, funding, depositing, creating factory instances, then the user action under test.
        - A semantic wrapper may execute a short valid lifecycle sequence inside one handler action when random interleavings are unlikely to reach the meaningful state. Keep the sequence generic and based on the target ABI, not on a single benchmark sample.
        - Prefer semantic stress sequences that are broadly useful:
          1. repeated settlement/distribution/claim/withdraw calls from a valid state,
          2. repeated mint/buy by the same allowed user when a per-user limit is implied,
          3. deposit/mint followed by withdraw/redeem with independent backing checks,
          4. factory/proxy creation followed by a real call through the created instance.
        - Keep these checks conservative. If an obligation is inferred but the observable state is insufficient, omit the assertion or log it via a non-failing helper; do not create a false positive.
        - Semantic invariants should include independent inequalities or reachability checks where possible, not only target-vs-ghost mirrors.
        - A pure ghost mirror is NOT enough for liability/backing/replay obligations. If the target exposes claimable balances or custody balances,
          include an independent invariant such as `sum(target.claims(trackedActors)) <= address(target).balance` or the token-backed equivalent.
        - For repeated settlement/distribution/finalization wrappers, do not update the ghost model to mirror duplicated target credit and stop there.
          Also assert that repeated execution cannot create more claimable liabilities than observable backing assets.
        - For lifecycle/cap properties, account for lazy state updates and monotonic time. A cap invariant tied to `currentState == PreSale` is unsafe if your handler can warp time backwards or leave state stale; prefer execution-based checks in the wrapper that performed the transition/invest, or omit the invariant when the precondition is not stable.
        - For mutable cap/limit properties, do not compare historical cumulative usage against a current cap that an admin setter may have lowered after the usage occurred. A valid cap test is: before a mint/buy/claim under the current cap, if `used + amount > currentCap`, the call must revert or leave `used` unchanged; if it succeeds, update ghost usage and assert the new usage is within the cap used for that call.

        [OUTPUT RULES]
        - Output valid, compilable Solidity ^0.8.20 code.
        - Do not use hardcoded addresses like `0xMcMc...`; use `makeAddr("alice")` or `vm.addr(1)`.
        - Output ONLY the code block.
        """

        user_prompt = f"""
        [TARGET CONTRACT TO ANALYZE]
        {target_code}

        [SAFETY RULES REFERENCE]
        {safety_code}

        [SEMANTIC OBLIGATIONS]
        {semantic_context if semantic_context else "None"}

        [INSTRUCTION]
        Generate a robust Handler-based invariant test suite (`Invariant.t.sol`) for the target above.
            1. Use the [CRITICAL] selector initialization pattern (dynamic array).
               The array size must exactly match the assigned handler selectors; no unassigned selector slots.
            2. Ensure strictly import from "./Target.sol".
            3. Implement optimistic ghost accounting with revert recovery.
               Prefer commit-on-success ghost accounting to avoid false positives from reverted calls.
            4. Avoid comparing uint with address in loops.
            5. `InvariantTest` must inherit only `Test`; do not import/use `StdInvariant` or `Console`.
            6. Avoid public-array/manual-getter name collisions.
                7. Do not use stale per-index snapshots for duplicate recipients; use unique recipients or post-success `+=` ghost updates.
                8. Include at most one handler wrapper per target function in `targetSelector`.
                9. For feasible SEMANTIC obligations, include at least one independent safety invariant or wrapper-local assertion.
                   Do not rely only on target-vs-ghost equality for solvency, backing, replay, or cap semantics.
                   For caps/limits that can be changed by owner/admin setters, prefer wrapper-local assertions around the user action, or keep the setter out of `targetSelector`; do not make historical usage fail after a later admin cap decrease.
                10. For independent balance-sum invariants, sum over a unique
                    actor/holder list exactly once. If `owner` is already inserted
                    into that list, do not add `handler.owner()` separately.
                11. Never write `target.fn{{value: 0}}(...)` for zero-payment calls; write `target.fn(...)`.
                    """

        print("   Architect is designing Generic Handler-Based Invariant Tests...")
        return self._sanitize_fuzz_test(self._query_llm(system_prompt, user_prompt), target_code)

    def _sanitize_fuzz_test(self, code: str, target_code: str = "") -> str:
        """Patch common invariant-harness patterns that create false positives."""
        source = self._normalize_solidity_code(code or "")
        source = self._normalize_payable_target_casts(source, target_code)
        source = self._strip_zero_value_call_options(source)
        source = self._normalize_erc721_mock_approval(source)
        counter = 0

        # LLMs sometimes generate unique-pick helpers like:
        #   uint256 stride = (seed % actors.length) + 1;
        #   while (idx < count) { ... pos = (pos + stride) % actors.length; }
        # If stride and length are not coprime, the loop may never collect
        # enough unique actors and Foundry appears to hang. Linear stride is
        # less random but always progresses when count <= actors.length.
        source = re.sub(
            r"uint256\s+stride\s*=\s*\([^;\n]*%\s*[A-Za-z_][A-Za-z0-9_]*\.length\s*\)\s*\+\s*1\s*;",
            "uint256 stride = 1; // LEVER: linear stride prevents unique-pick fuzz hangs;",
            source,
        )
        source_lower = source.lower()
        has_historical_cap_invariant = (
            ("cap" in source_lower or "limit" in source_lower)
            and any(term in source_lower for term in ("minted", "claimed", "deposited", "withdrawn", "used"))
            and "assertle" in source_lower
        )

        def is_mutable_policy_selector(rhs: str) -> bool:
            if not has_historical_cap_invariant:
                return False
            normalized = re.sub(r"\s+", "", rhs)
            return bool(
                re.search(
                    r"\.(?:h_)?(?:set|update|change)[A-Za-z0-9_]*(?:Cap|Limit|Max|Threshold)[A-Za-z0-9_]*\.selector\b",
                    normalized,
                )
            )

        def repl_warp(match: re.Match) -> str:
            nonlocal counter
            indent = match.group("indent")
            expr = match.group("expr").strip()
            counter += 1
            var_name = f"__leverWarpTarget{counter}"
            return (
                f"{indent}uint256 {var_name} = {expr};\n"
                f"{indent}if ({var_name} > block.timestamp) {{\n"
                f"{indent}    vm.warp({var_name});\n"
                f"{indent}}}"
            )

        # Fuzzing lifecycle contracts with backwards time travel creates
        # impossible stale-state counterexamples. Keep all generated warps
        # monotonic while preserving forward time exploration.
        pattern = re.compile(
            r"^(?P<indent>\s*)vm\.warp\((?P<expr>[^;\n]+)\);",
            flags=re.MULTILINE,
        )
        source = pattern.sub(repl_warp, source)

        # A common false positive is: Handler._pushUnique(owner) inserts owner
        # into actorAt(i), then an invariant sums actorAt(i) and adds owner again.
        # That double-counts owner and can make sum(tracked balances) exceed supply
        # even when target and ghost accounting agree.
        if re.search(r"_pushUnique\s*\(\s*owner\s*\)", source):
            source = re.sub(
                r"(?m)^(?P<indent>\s*)sum\s*\+=\s*([A-Za-z_][A-Za-z0-9_]*\.)?balanceOf\s*\(\s*handler\.owner\s*\(\s*\)\s*\)\s*;",
                r"\g<indent>// LEVER: owner is already in the unique actor list; avoid double-counting.",
                source,
            )
            source = re.sub(
                r"(?m)^(?P<indent>\s*)sum\s*=\s*sum\s*\+\s*([A-Za-z_][A-Za-z0-9_]*\.)?balanceOf\s*\(\s*handler\.owner\s*\(\s*\)\s*\)\s*;",
                r"\g<indent>// LEVER: owner is already in the unique actor list; avoid double-counting.",
                source,
            )

        def repl_selector_block(match: re.Match) -> str:
            decl = match.group("decl")
            name = match.group("name")
            body = match.group("body")

            assign_pattern = re.compile(
                rf"^(?P<indent>\s*){re.escape(name)}\s*\[\s*\d+\s*\]\s*=\s*(?P<rhs>[^;\n]+);\s*$",
                flags=re.MULTILINE,
            )

            rhs_values = []
            for assign in assign_pattern.finditer(body):
                rhs = assign.group("rhs").strip()
                if rhs in {"bytes4(0)", "0x00000000", "bytes4(0x0)"}:
                    continue
                if is_mutable_policy_selector(rhs):
                    continue
                rhs_values.append(rhs)

            if not rhs_values:
                return match.group(0)

            next_index = 0

            def repl_assignment(assign: re.Match) -> str:
                nonlocal next_index
                indent = assign.group("indent")
                rhs = assign.group("rhs").strip()
                if rhs in {"bytes4(0)", "0x00000000", "bytes4(0x0)"}:
                    return f"{indent}// LEVER: removed zero selector from fuzz target list"
                if is_mutable_policy_selector(rhs):
                    return f"{indent}// LEVER: removed mutable cap/limit setter from fuzz target list to avoid retroactive-policy false positives"
                rewritten = f"{indent}{name}[{next_index}] = {rhs};"
                next_index += 1
                return rewritten

            new_decl = re.sub(
                r"new\s+bytes4\[\]\(\s*\d+\s*\)",
                f"new bytes4[]({len(rhs_values)})",
                decl,
                count=1,
            )
            new_body = assign_pattern.sub(repl_assignment, body)
            return new_decl + new_body

        selector_block_pattern = re.compile(
            r"(?P<decl>bytes4\[\]\s+memory\s+(?P<name>[A-Za-z_]\w*)\s*=\s*new\s+bytes4\[\]\(\s*\d+\s*\)\s*;\s*\n)"
            r"(?P<body>.*?targetSelector\s*\(\s*FuzzSelector\s*\(\s*\{[^{}]*selectors\s*:\s*(?P=name)[^{}]*\}\s*\)\s*\)\s*;)",
            flags=re.DOTALL,
        )
        source = selector_block_pattern.sub(repl_selector_block, source)
        if (
            "putApesToAgency" in source
            and "setApprovalForAll" in source
            and "target.putApesToAgency" in source
            and not re.search(r"\.setApprovalForAll\s*\(\s*address\s*\(\s*target\s*\)", source)
        ):
            source = re.sub(
                r"(?P<indent>\s*)vm\.startPrank\((?P<actor>[^;\n]+)\);\s*(?P<tryindent>\s*)try\s+target\.putApesToAgency\(",
                (
                    r"\g<indent>vm.startPrank(\g<actor>);\n"
                    r"\g<tryindent>try nft.setApprovalForAll(address(target), true) { } catch { }\n"
                    r"\g<tryindent>try target.putApesToAgency("
                ),
                source,
            )
        return self._strip_zero_value_call_options(self._normalize_payable_target_casts(source, target_code))

    def _sanitize_semantic_probe_test(self, code: str) -> str:
        """Patch common Foundry one-shot prank mistakes in generated probes."""
        source = code or ""
        source = self._normalize_erc721_mock_approval(source)
        source = re.sub(r"\bassertEq0\b", "_leverAssertBytesEq", source)
        source = self._strip_zero_value_call_options(source)

        # `vm.expectRevert` and low-level `.call` are mutually exclusive ways to
        # assert rejection. Foundry consumes the expected revert and can make the
        # low-level call appear successful, so the following `!ok` assertion then
        # reports a false failure. Keep the low-level `(bool ok, ...)` assertion
        # and remove only the immediately preceding expectRevert cheatcode.
        low_level_revert_pattern = re.compile(
            r"^[ \t]*vm\.expectRevert\([^;\n]*\);[ \t]*\n"
            r"(?P<prank>[ \t]*vm\.(?:prank|startPrank)\([^;\n]*\);[ \t]*\n)?"
            r"(?=[ \t]*\(bool[^;\n]*\)\s*=\s*[^;\n]*\.call)",
            flags=re.MULTILINE,
        )
        source = low_level_revert_pattern.sub(lambda match: match.group("prank") or "", source)
        counter = 0

        def repl_change_params(match: re.Match) -> str:
            nonlocal counter
            actor = match.group("actor").strip()
            expect = match.group("expect") or ""
            cap_expr = match.group("cap").strip()
            rate_expr = match.group("rate").strip()
            counter += 1
            cap_name = f"__leverCachedMaxPresaleSupply{counter}"
            return (
                f"uint256 {cap_name} = {cap_expr};\n"
                f"        vm.prank({actor});\n"
                f"        {expect}"
                f"target.changeInvestmentParameters({cap_name}, {rate_expr});"
            )

        pattern = re.compile(
            r"vm\.prank\((?P<actor>[^;]+)\);\s*"
            r"(?P<expect>(?:vm\.expectRevert\([^;]*\);\s*)?)"
            r"target\.changeInvestmentParameters\(\s*"
            r"(?P<cap>target\.maxPresaleSupply\(\)(?:\s*[+\-]\s*[^,\n;)]+)?)\s*,\s*"
            r"(?P<rate>[^;\n]+?)\s*\);",
            flags=re.MULTILINE,
        )
        source = pattern.sub(repl_change_params, source)

        # Semantic probes are meant to fail when they demonstrate a requirement
        # violation. Models sometimes write a "diagnostic" test that asserts the
        # bad condition itself, causing Foundry to mark the vulnerable behavior as
        # PASS. Flip only clearly unsafe semantic assertions into safety checks.
        unsafe_words = (
            r"exceed|exceeding|duplicate|duplicated|replay|unbacked|underbacked|"
            r"violat|unsafe|insolven|bypass|without backing|without source"
        )

        def repl_unsafe_assert_gt(match: re.Match) -> str:
            left = match.group("left").strip()
            right = match.group("right").strip()
            msg = match.group("msg")
            return f'assertLe({left}, {right}, "{msg}");'

        source = re.sub(
            rf'assertGt\s*\(\s*(?P<left>[^,;\n]+?)\s*,\s*(?P<right>[^,;\n]+?)\s*,\s*"(?P<msg>SEMANTIC:[^"]*(?:{unsafe_words})[^"]*)"\s*\)\s*;',
            repl_unsafe_assert_gt,
            source,
            flags=re.IGNORECASE,
        )
        return source

    def generate_semantic_probe_test(
        self,
        target_code: str,
        safety_code: str,
        semantic_context: str,
        error_feedback: str = "",
        previous_code: str = "",
    ) -> str:
        feedback_section = ""
        if error_feedback:
            feedback_section = f"""
            [PREVIOUS SEMANTIC PROBE COMPILE/RUN ERROR]
            Your previous SemanticProbe.t.sol failed before it could provide useful diagnostic evidence.
            ERROR LOG:
            {error_feedback}

            [PREVIOUS SEMANTICPROBE.T.SOL]
            {previous_code if previous_code else "(not provided)"}

            [INSTRUCTION]
            Output a complete replacement `SemanticProbe.t.sol`. Repair the probe only; do not change target semantics.
            """

        system_prompt = f"""
        You are a Foundry Semantic Probe Architect.
        Generate a deterministic `SemanticProbe.t.sol` diagnostic test suite for requirement-derived semantic obligations.

        {feedback_section}

        [PURPOSE]
        These probes are NOT the paper-facing Foundry invariant metric. They are lower-confidence diagnostic tests
        that exercise positive lifecycle paths and expose semantic evidence that random fuzzing may miss.

        [FILE AND IMPORT RULES]
        - The target contract is always in `./Target.sol`.
        - Import Foundry as `import "forge-std/Test.sol";`.
        - Import SafetyRules only if it is useful: `import "./SafetyRules.sol";`.
        - Do not import `StdInvariant`, `Console`, `Script`, or external files that are not present.
        - Output exactly one Solidity file using pragma `^0.8.20`.

        [TEST SHAPE]
        - Implement `contract SemanticProbeTest is Test`.
        - Use deterministic `function test_semantic_...() public` test methods.
        - Deploy the target in `setUp()` when constructor arguments are simple. If the target needs mocks, create minimal local mocks in the same file only when required by the constructor.
        - Create explicit actors (`owner`, `alice`, `bob`, `mallory`) with `makeAddr`.
        - Fund every actor and the target when needed using `vm.deal`.
        - When a call must come from an actor, cache any needed view values before `vm.prank(actor)` because a one-shot prank is consumed by the next external call.
          Never write `vm.prank(owner); target.adminCall(target.getter(), x);`: the getter consumes the prank and the admin call will come from the test contract.
          Instead write `uint256 v = target.getter(); vm.prank(owner); target.adminCall(v, x);`.
          If you need several calls from the same actor, use `vm.startPrank(actor); ... vm.stopPrank();`.
        - Keep authorization identities consistent. If the constructor stores `msg.sender` as owner/governance/admin, deploy under the intended actor with `vm.prank(actor)` or use the actual public getter/constructor argument as the authorized caller. Do not assume `owner == governance` unless the target code proves it.
        - Pre-create helper contracts and compute function arguments before `vm.expectRevert(...)`; Foundry applies `expectRevert` to the next external call/create, so `target.fn(address(new Helper()))` is unsafe.
          The same applies after `vm.prank(...)`: no target getter, helper creation, or external argument expression may appear before the intended actor call.
        - If a revert type is uncertain or uses a custom error with arguments, prefer `vm.expectRevert()` or the exact selector/encoding from the target code. Do not fail a semantic test only because the revert encoding differs while the required revert happened.
        - When a call is expected to be rejected at a boundary where several guards may apply, such as cap exceeded versus wrong payment, use `vm.expectRevert()` instead of asserting one exact revert string. The semantic fact is rejection, not guard ordering.
        - Never combine `vm.expectRevert(...)` with a low-level `.call(...)` whose `(bool ok, bytes data)` result is inspected. For low-level calls, do not use `expectRevert`; assert `ok == false` and inspect the returned data if needed. For typed calls that bubble reverts, use `expectRevert`.

        [SEMANTIC OBJECTIVES]
        Use the provided [SEMANTIC OBLIGATIONS] as goals. Prefer broadly reusable probes:
        - Positive-path reachability: initialize/open/activate phases with the authorized actor before judging behavior.
        - Replay/consumption: repeat distribution, claim, settlement, or withdraw after a successful first call and check for duplicated credit or missing consumption.
        - Liability solvency: compare exposed claimable/accounting liabilities for tracked actors against actual ETH/token assets held by the target.
        - Per-user cap/allowlist: repeat mint/buy by the same authorized user and check whether a per-user limit is actually enforced when the requirement implies one.
        - Asset backing: deposit/mint should correspond to observable ETH/token custody; withdraw/redeem should release backing when the interface exposes enough evidence.
        - Proxy/factory usability: created instances must have nonzero runtime code and support at least one expected method call.

        [CONFIDENCE RULES]
        - Do not invent exact numeric caps, private state, or benchmark-specific constants. Derive amounts from public values or use small generic quantities.
        - Respect Solidity integer units exactly. Compute expected payouts/remainders as `per = amount / n` and `remainder = amount - per * n`; do not assume ether-denominated values divide into whole ether.
        - For replay/consumption probes, treat either an explicit revert or an idempotent no-op as safe when no new entitlement/assets were added. Fail only if repeated execution duplicates credit, drains extra assets, or violates a concrete backing/accounting invariant.
        - For allowance-based flows such as `approve`/`transferFrom`, a repeated `transferFrom` is normally valid while sufficient allowance and balance remain. Compute the exact remaining allowance and balance after each successful call, assert monotonic consumption/accounting, and only expect a revert when the next amount exceeds the remaining allowance or balance.
        - A semantic probe must encode the expected SAFE property. Never assert that the unsafe condition is true.
          Wrong: `assertGt(liabilities, assets, "SEMANTIC: duplicated credit exceeds assets")`.
          Correct: `assertLe(liabilities, assets, "SEMANTIC: duplicated credit exceeds assets")`.
          If you intentionally observe a violation, make the test fail with the safe assertion or `fail("SEMANTIC: ...")`.
        - Do not fail only because an optional standard function is absent. Fail only after a concrete positive-path behavior demonstrates semantic evidence.
        - If the semantic evidence is ambiguous, make the test non-failing and document the observation with a short assertion message or comments.
        - If the positive path cannot be reached after reasonable setup, prefer `assertTrue(true, "... low-confidence unreachable positive path")` over a false failure.
        - Assertion messages for true semantic evidence should start with `SEMANTIC:` so downstream logs can classify them.

        [OUTPUT RULES]
        - Output valid, compilable Solidity only.
        - No Markdown fences.
        - Keep the probe compact and deterministic.
        - Do not define helpers named `assertEq0`; that name collides with forge-std internals. Use `_assertBytesEq` or compare `keccak256` hashes for dynamic bytes/string data.
        - Do not attach `{{value: 0}}` to nonpayable calls. Use plain calls when no ETH is being sent.
        """

        user_prompt = f"""
        [TARGET CONTRACT]
        {target_code}

        [SAFETY RULES REFERENCE]
        {safety_code}

        [SEMANTIC OBLIGATIONS]
        {semantic_context if semantic_context else "None"}

        [INSTRUCTION]
        Generate `SemanticProbe.t.sol` now.
        """

        print("   Architect is designing Semantic Probe Tests...")
        return self._sanitize_semantic_probe_test(self._query_llm(system_prompt, user_prompt))

# ==========================================
# 6. Attack Analyst
# ==========================================
class AttackAnalyst(BaseAgent):
    """
    Analyzes execution logs from the Simulator.
    Generates specific Constraints (for Coder) and Properties (for Verifier) to fix vulnerabilities.
    """
    def analyze_attack(self, logs: str) -> str:
        system_prompt = """
        You are a local smart-contract security diagnostic analyst.
        You are provided with local verification logs where a simulated safety violation was observed.

        [TASK]
        1. Analyze the logs to identify the root cause (e.g., Reentrancy, Integer Overflow, Slippage, Logic Error).
        2. Formulate a [CONSTRAINT] for the Solidity Coder to fix the code.
        3. Formulate a [PROPERTY] for the Formal Verifier to ensure this specific bug never happens again.

        [OUTPUT FORMAT]
        You must output specific sections identified by tags.

        [ANALYSIS]
        ... brief explanation of the local safety violation ...

        [CONSTRAINT]
        ... specific instruction to fix the code (e.g., "Add require(x > 0)") ...

        [PROPERTY]
        ... formal property description (e.g., "Solvency: The contract balance must not decrease below initial deposits") ...
        """

        user_prompt = f"""
        [LOCAL DIAGNOSTIC LOGS]
        {logs}

        [INSTRUCTION]
        Provide the Analysis, Constraint, and new Property.
        """

        print("   AttackAnalyst is determining root cause...")
        return self._query_llm(system_prompt, user_prompt)

    def analyze_attack_structured(self, logs: str) -> Dict:
        system_prompt = """
        You are a local smart-contract security diagnostic analyst for the LeVer framework.
        You receive local simulation and fuzzing logs, including the scheduler phase
        and pending honest intent when available. The goal is defensive verification of
        generated/local contracts, not guidance for real-world exploitation.

        [TASK]
        Produce a Diagnostic-to-Property feedback object. The property must be precise enough
        for a formal verifier to turn it into a theorem and for a Solidity coder to repair
        the root cause.

            [ROOT-CAUSE DISCIPLINE]
        1. Prefer the concrete Foundry failing invariant, shrunk sequence, and trace over
           broad guesses from function names.
        2. If the log contains a `[FOUNDRY STRUCTURED SUMMARY]`, use it as the primary
           evidence for:
           - failing invariant name,
           - passed/failed test counts,
           - shrunk sequence,
           - trace excerpt.
        3. Classify the root cause using this taxonomy:
           - Access-control failure: a caller without the required role can perform a
             privileged state transition, configuration change, mint/burn/claim update, or asset movement.
           - State-machine failure: a transition reaches a phase or terminal condition that
             violates the intended lifecycle.
           - Accounting/liability failure: claims, balances, shares, or liabilities can grow
             without matching backing assets, or assets can leave without reducing the matching claim.
           - Idempotency/replay failure: repeating a settlement/finalization/distribution-like
             operation duplicates effects that should be one-time or monotonic.
           - External-call/order failure: value transfer or external calls can observe or exploit
             partially updated state, unchecked failure, or inconsistent rollback.
           - Input-domain failure: malformed, zero, duplicate, or mismatched inputs create invalid state.
           - Harness/modeling failure: ghost state changed on revert, duplicate users are
             mishandled, or the invariant tracks state not actually modeled or exposed.
        4. Do not blame a function merely because it appears in logs. A function is causal
           only if the trace shows a successful effect from it, or the failing invariant
           logically requires that effect.
        5. When a trace has repeated successful calls of the same effect class, decide
           whether the intended semantics are repeatable, one-shot, or monotonic, and state
           the repair in those general terms.
        6. When a trace reaches a sensitive phase or asset-moving path, identify the required
           caller role from the requirement or constructor state and include that authorization
           rule in the repair constraint.
        7. Prioritize requirement-derived semantic obligations over generic standards findings.
           If logs contain concrete execution evidence for a business semantic issue
           (cap/limit bypass, unbacked mint/deposit, duplicated distribution/claim,
           broken positive path, unusable proxy/clone), diagnose that issue first even
           when an ERC/interface/compliance invariant is also failing. Diagnose a generic
           ERC/interface issue first only when there is no stronger requirement-grounded
           semantic evidence, or when that interface issue directly blocks the main
           positive path.
        8. If evidence points to the harness rather than the target, return a property with
               `"status": "harness_issue"` and a constraint that fixes the harness, not the contract.
        9. In Foundry invariant logs, the `sender=` in a shrunk sequence is usually the
               caller of the Handler, not necessarily `msg.sender` seen by the Target. If the
               Handler wrapper uses `vm.prank(owner)`, `vm.startPrank(owner)`, or another
               explicit actor before calling the Target, treat the Target call as being made by
               that pranked actor. Do not classify an access-control failure solely because the
               shrunk sequence sender is not the owner.
        10. If a failing invariant compares Target state to Handler ghost state, first check
               whether the Handler has duplicate-recipient handling, revert-only commit, and one
               wrapper per Target function. If these are violated, classify as `harness_issue`
               unless there is an independent non-ghost semantic invariant or direct Target trace
               showing the same bug.
        11. Treat semantic-probe failures as harness issues when the evidence is caused by:
               wrong deployer/owner/governance identity, a mismatched custom-error selector while
               the required revert did occur, `vm.expectRevert` being consumed by helper contract
               creation or argument construction, or incorrect expected arithmetic units/remainders.
               Only classify a target bug when the trace shows the target successfully violating
               the semantic obligation after valid setup.
        12. Treat mutable-policy invariant failures as harness issues when the shrunk sequence
           first performs successful historical usage, then an authorized admin/owner setter lowers
           a cap, limit, threshold, max, price, or rate, and the invariant fails only because it
           compares past usage against the new current policy. That is not a cap bypass unless a
           later user action succeeds despite `used + amount > currentCap`. In this case return
           `"status": "harness_issue"` and constrain the harness to use wrapper-local cap checks,
           keep the policy non-decreasing, or omit the mutable setter from the fuzz selector.

        [STRICT OUTPUT]
        Return ONLY valid JSON with this shape:
        {
          "analysis": "brief root-cause explanation",
          "constraint": "specific Solidity repair instruction",
          "property": {
            "source_trace": "short trace/counterexample summary",
            "property_text": "security property to enforce",
            "invariant_template": "candidate invariant or theorem template",
            "status": "new"
          }
        }
        """

        user_prompt = f"""
        [LOCAL DIAGNOSTIC / FUZZ LOGS]
        {logs}

        [INSTRUCTION]
        Return the JSON object only.
        """

        print("   AttackAnalyst is producing structured diagnostic-to-property feedback...")
        response = self._query_llm(system_prompt, user_prompt)
        try:
            return json.loads(response)
        except Exception:
            match = re.search(r"\{[\s\S]*\}", response)
            if match:
                try:
                    return json.loads(match.group(0))
                except Exception:
                    pass
            return {
                "analysis": "Failed to parse structured analyst output.",
                "constraint": "",
                "property": {
                    "source_trace": logs[:1000],
                    "property_text": response.strip(),
                    "invariant_template": "",
                    "status": "parse_failed",
                },
                "raw": response,
            }

    def _parse_semantic_suspicion_response(self, response: str) -> Dict:
        """Parse AttackAnalyst JSON, preserving suspicious=true on minor JSON mistakes."""
        raw = (response or "").strip()
        candidates = []
        if raw:
            candidates.append(raw)
        match = re.search(r"\{[\s\S]*\}", raw)
        if match:
            candidates.append(match.group(0).strip())

        repaired_candidates = []
        for candidate in candidates:
            repaired_candidates.append(candidate)
            # Common model slip: close the evidence array with `}` just before
            # `coverage_gap`. Repair only this narrow shape so unrelated JSON is
            # not silently rewritten.
            repaired_candidates.append(re.sub(
                r'("evidence"\s*:\s*\[[\s\S]*?)\n\s*}\s*,\s*\n\s*("coverage_gap"\s*:)',
                r'\1\n  ],\n  \2',
                candidate,
                count=1,
            ))

        seen = set()
        for candidate in repaired_candidates:
            if not candidate or candidate in seen:
                continue
            seen.add(candidate)
            try:
                data = json.loads(candidate)
                if isinstance(data, dict):
                    return data
            except Exception:
                pass

        return self._salvage_semantic_suspicion_response(raw)

    def _salvage_semantic_suspicion_response(self, response: str) -> Dict:
        """Best-effort field extraction when the model returns nearly-JSON text."""
        raw = response or ""
        lower = raw.lower()
        suspicious = bool(re.search(r'"suspicious"\s*:\s*true\b', raw, flags=re.IGNORECASE))

        def extract_string(key: str) -> str:
            match = re.search(
                rf'"{re.escape(key)}"\s*:\s*"((?:\\.|[^"\\])*)"',
                raw,
                flags=re.DOTALL,
            )
            if not match:
                return ""
            try:
                return json.loads(f'"{match.group(1)}"')
            except Exception:
                return match.group(1).replace('\\"', '"')

        evidence = []
        evidence_block = re.search(
            r'"evidence"\s*:\s*\[([\s\S]*?)(?:\]\s*,|\}\s*,\s*"coverage_gap")',
            raw,
        )
        if evidence_block:
            evidence = [
                item
                for item in re.findall(r'"((?:\\.|[^"\\])*)"', evidence_block.group(1))
                if item.strip()
            ]

        property_text = extract_string("property_text")
        invariant_template = extract_string("invariant_template")
        source_trace = extract_string("source_trace")
        if suspicious and not property_text and any(token in lower for token in [
            "unbacked",
            "without observable",
            "without custody",
            "without paying out",
            "no eth payout",
        ]):
            property_text = (
                "Deposit/mint credit must be backed by observable asset inflow "
                "or authorized custody movement, and withdraw/burn must release "
                "the corresponding backing asset."
            )
            invariant_template = invariant_template or (
                "Successful deposit/mint increases user credit only with matching "
                "asset inflow or custody consumption; successful withdraw/burn "
                "must pay out or release backing to the caller."
            )
            source_trace = source_trace or (
                "Semantic evidence reported successful deposit/withdraw accounting "
                "changes without visible custody or payout movement."
            )

        confidence = extract_string("confidence")
        if suspicious and confidence.lower() not in {"low", "medium", "high"}:
            confidence = "medium"

        return {
            "suspicious": suspicious,
            "confidence": confidence,
            "analysis": extract_string("analysis"),
            "evidence": evidence,
            "coverage_gap": extract_string("coverage_gap"),
            "constraint": extract_string("constraint") or property_text,
            "property": {
                "source_trace": source_trace,
                "property_text": property_text,
                "invariant_template": invariant_template,
                "status": extract_string("status") or ("suspicious_unconfirmed" if suspicious else ""),
            } if suspicious and (property_text or invariant_template or source_trace) else {},
        }

    def analyze_semantic_suspicion(self, logs: str) -> Dict:
        system_prompt = """
        You are the semantic evidence analyst for the LeVer framework.
        You are called only when the hard dynamic checks did NOT confirm a bug:
        agent safety checks and Foundry invariants appear clean, or the observed
        action was not counted as a confirmed breach.

        [TASK]
        Decide whether the logs still contain credible semantic evidence that the
        contract may violate the requirement and therefore deserves another
        iteration with a stronger property/harness.

        [WHAT COUNTS AS SUSPICIOUS]
        Mark suspicious=true only when there is concrete evidence such as:
        - a sensitive call succeeded and its observed effect contradicts a
          requirement-derived semantic obligation;
        - repeated successful calls by the same actor appear to bypass a cap,
          one-shot, consumption, or idempotency requirement;
        - a deposit/mint/share-credit path succeeds without observable backing
          asset movement, or withdraw/redeem burns accounting without returning
          backing assets;
        - a factory/proxy/deployment path returns an address that may have no
          usable runtime code, cannot be initialized, or cannot be called as the
          required instance;
        - SafetyRules or Foundry marked a semantic obligation as N/A even though
          agent traces contain enough execution evidence to test it dynamically
          AND the dynamic logs show a possible requirement violation or an
          untested suspicious effect that was not already covered by a passing
          semantic probe;
        - an action script is marked ACTION_PARTIAL_EFFECT_OBSERVED, or includes
          both a successful suspicious operation and a later expected-revert
          probe, causing the whole action to be treated as failed even though
          the suspicious effect occurred in the trace.
        - a semantic probe source contains a test that asserts an unsafe condition
          as the success case, e.g. an `assertGt` whose SEMANTIC message says
          liabilities exceed assets, duplicated credit was created, backing is
          missing, or a cap was bypassed. That is still credible semantic
          evidence even if Foundry reports the test as PASS.

        [PRIORITY]
        Requirement-derived semantic obligations outrank generic standard/compliance
        observations. If the logs show concrete business evidence such as a cap/limit
        bypass, unbacked mint/deposit, duplicated distribution/claim, broken positive
        path, or unusable factory/proxy instance, mark that semantic issue suspicious
        even if another ERC/interface issue is also present. Treat ERC/interface
        compliance as the primary issue only when no stronger semantic evidence exists
        or when it directly blocks the main positive path.

        [WHAT DOES NOT COUNT]
        Do not mark suspicious for mere failed probes, reverted unauthorized
        calls, missing logs alone, generic best-practice concerns, or a property
        that is not supported by the requirement text or selected semantic
        obligations. Do not invent a numeric cap unless the requirement provides
        one; express the property as per-user monotonic/capped behavior.
        Do not mark suspicious merely because a view-only SafetyRules check
        returns N/A when Foundry fuzzing and semantic probes already executed
        the corresponding positive and negative paths and passed. Coverage gaps
        alone are useful notes, but they should not force another contract
        iteration unless there is a concrete unsafe effect or a missing dynamic
        check for an observed suspicious behavior.
        Treat owner withdrawal of collected sale/mint proceeds as ordinary
        revenue handling, not as a liability-solvency violation, unless the
        requirement says users retain refundable claims or the trace shows
        withdrawals can consume assets backing outstanding claims.
        Treat Foundry harness lifecycle mistakes as non-suspicious harness issues,
        especially when logs show `VM::prank(actor)` followed by a target getter or
        other external call before the intended privileged call: one-shot prank was
        consumed by the getter, so a later `Not owner`/authorization revert is not
        evidence about the contract. Public lazy lifecycle refresh functions such as
        `transitionState()` are also not suspicious unless a requirement says they
        are privileged or the trace shows a concrete phase change that blocks or
        enables an otherwise forbidden value-moving operation.
        Do not dismiss a probe merely because it is diagnostic-only. If the source
        or logs show the probe deliberately reached a valid positive path and then
        observed a semantic violation, mark suspicious and propose the corresponding
        general safety property.

        [OUTPUT]
        Return ONLY valid JSON:
        {
          "suspicious": true,
          "confidence": "low|medium|high",
          "analysis": "brief explanation grounded in observed traces",
          "evidence": ["short concrete evidence item"],
          "coverage_gap": "what SafetyRules/Foundry failed to check",
          "constraint": "specific general instruction for the next iteration",
          "property": {
            "source_trace": "short trace summary",
            "property_text": "semantic property to enforce",
            "invariant_template": "candidate dynamic invariant/theorem template",
            "status": "suspicious_unconfirmed"
          }
        }

        If evidence is weak, return suspicious=false with empty constraint and
        empty property.
        The JSON must be parseable: close arrays with `]`, close objects with `}`,
        escape embedded quotes, and do not include comments or Markdown fences.
        """

        user_prompt = f"""
        [CLEAN-RUN LOGS WITH POSSIBLE SEMANTIC EVIDENCE]
        {logs}

        [INSTRUCTION]
        Return the JSON object only.
        """

        print("   AttackAnalyst is checking for unconfirmed semantic evidence...")
        response = self._query_llm(system_prompt, user_prompt)
        data = self._parse_semantic_suspicion_response(response)
        if not isinstance(data, dict):
            data = {}
        data.setdefault("suspicious", False)
        data.setdefault("confidence", "")
        data.setdefault("analysis", "")
        data.setdefault("evidence", [])
        data.setdefault("coverage_gap", "")
        data.setdefault("constraint", "")
        data.setdefault("property", {})
        if data.get("suspicious"):
            joined = json.dumps(data, ensure_ascii=False).lower()
            logs_lower = (logs or "").lower()
            semantic_probe_clean = "semantic_probe" in logs_lower and "suite result: ok." in logs_lower and "[fail:" not in logs_lower
            na_coverage_only = "n/a" in joined and ("coverage gap" in joined or "safetyrules" in joined)
            normal_revenue_pattern = any(token in joined for token in [
                "collected payments",
                "owner withdraw",
                "owner withdrawals",
                "withdraw drains",
                "withdrawal drains",
                "sale proceeds",
                "mint proceeds",
            ])
            concrete_bad_pattern = any(token in joined for token in [
                "cap bypass",
                "bypass a cap",
                "duplicated credit",
                "liabilities exceed",
                "claims exceed",
                "unbacked",
                "underbacked",
                "without backing",
                "no runtime code",
                "unusable",
                "positive path cannot",
                "unsafe condition",
                "assertgt",
            ])
            if semantic_probe_clean and na_coverage_only and normal_revenue_pattern and not concrete_bad_pattern:
                data["suspicious"] = False
                data["confidence"] = "low"
                data["analysis"] = (
                    "Filtered as non-suspicious: the only issue is a view-only "
                    "SafetyRules N/A/coverage note, while dynamic semantic probes "
                    "passed the corresponding sale-payment and withdrawal behavior."
                )
                data["coverage_gap"] = data.get("coverage_gap", "")
                data["constraint"] = ""
                data["property"] = {}
        return data

class ResultSummarizer(BaseAgent):
    def summarize_experiment(self, contract_name, requirement, metrics, fuzz_log, agent_log):
        system_prompt = """
        You are a Lead Smart Contract Auditor.
        Your job is to generate a structured "Post-Mortem Report" for a security experiment on a generated contract.

        [INPUTS]
        1. **Requirement**: What the contract was supposed to do.
        2. **Metrics**: Quantitative data (Did Fuzzer break invariants? Did Agent claim success?).
        3. **Logs**: Snippets of the execution.

        [OUTPUT FORMAT - MARKDOWN]
        # Security Report: {contract_name}

        ## 1. Executive Verdict
        **Status**: [SECURE / VULNERABLE / BROKEN]
        **Confidence**: [High/Medium/Low]

        ## 2. Vulnerability Analysis
        *Briefly analyze what went wrong based on the logs.*
        - **Agent Findings**: Did the Agent exploit a logic flaw (e.g. public mint)?
        - **Fuzzer Findings**: Did the Fuzzer break a mathematical invariant (e.g. solvency)?

        ## 3. Framework Effectiveness
        *Evaluate which part of the system caught the bug.*
        - Did Agent and Fuzzer agree?
        - Did one catch what the other missed? (Complementarity Analysis)

        ## 4. Final Conclusion
        *One sentence summary of the code quality.*
        """

        # Truncate logs to avoid exceeding the model context limit.
        user_prompt = f"""
        [TARGET]: {contract_name}
        [REQUIREMENT]: {requirement[:600]}...

        [METRICS]:
        {json.dumps(metrics, indent=2)}

        [AGENT LOGS (Snippet)]:
        {agent_log[-10000:]}

        [FUZZ LOGS (Snippet)]:
        {fuzz_log[-2000:]}

        [TASK]
        Generate the report.
        """

        print(f"   Summarizing results for {contract_name}...")
        return self._query_llm(system_prompt, user_prompt)
