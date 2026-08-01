import os
import subprocess
import shutil
import re
import json
import random
import time
import socket
import atexit
import shlex
import urllib.request
from urllib.parse import urlparse
from typing import Dict, List, Tuple, Optional
from openai import OpenAI
from core.agents import API_KEY, BASE_URL, DEFAULT_MODEL_NAME, FLASH_MODEL_NAME, FatalLLMError, is_fatal_llm_error
from core.config import ANVIL_CONFIG # Local Anvil configuration.
from core.structured_logging import CommandRecorder, JsonlEventLogger, parse_foundry_output, text_tail

class SimulationRunner:
    def __init__(self, root_dir="simulation_env", attack_rounds: Optional[int] = None):
        self.root_dir = os.path.abspath(root_dir)
        marker_path = os.path.join(self.root_dir, ".lever_workspace")
        if os.path.exists(self.root_dir) and not os.path.isfile(marker_path):
            raise RuntimeError(
                f"Refusing to use unmarked simulation directory: {self.root_dir}"
            )
        os.makedirs(self.root_dir, exist_ok=True)
        if not os.path.exists(marker_path):
            with open(marker_path, "w", encoding="utf-8") as f:
                f.write("LeVer managed simulation workspace\n")
        self.src_dir = os.path.join(self.root_dir, "src")
        self.script_dir = os.path.join(self.root_dir, "script")
        self.action_archive_dir = os.path.join(self.root_dir, "action_scripts_archive")
        resolved_api_key = API_KEY or os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=resolved_api_key, base_url=BASE_URL) if resolved_api_key else None
        self.realtime_log_path = os.path.join(self.root_dir, "realtime_simulation.log")

        # Load local RPC and account settings.
        self.RPC_URL = ANVIL_CONFIG["rpc_url"]
        self.accounts_config = ANVIL_CONFIG["accounts"]
        self.attack_rounds = attack_rounds or int(os.getenv("LEVER_ATTACK_ROUNDS", "3"))
        self.command_history: List[Dict] = []
        self.event_logger: Optional[JsonlEventLogger] = None
        self.command_recorder = CommandRecorder(os.path.join(self.root_dir, "command_logs"))
        self._owned_anvil_process: Optional[subprocess.Popen] = None
        self._owned_anvil_log = None
        self._owned_anvil_cache_dir = os.path.join(self.root_dir, "anvil_cache")
        self._core_source_snapshot: Dict[str, str] = {}

        # Build the configured account-key list dynamically.
        self.ANVIL_KEYS = [data["private_key"] for data in self.accounts_config.values()]

    def _llm_completion_kwargs(self, model_name: str) -> Dict:
        kwargs: Dict = {}
        if (model_name or "").lower().startswith("gpt-5"):
            kwargs["max_completion_tokens"] = int(os.getenv("LLM_MAX_COMPLETION_TOKENS", "12000"))
            reasoning_effort = os.getenv("LLM_REASONING_EFFORT", "low")
            if reasoning_effort:
                kwargs["reasoning_effort"] = reasoning_effort
        request_timeout = os.getenv("LLM_REQUEST_TIMEOUT")
        if request_timeout:
            kwargs["timeout"] = float(request_timeout)
        return kwargs

    def _chat_completion_with_retries(self, purpose: str, **kwargs):
        max_retries = int(os.getenv("LLM_MAX_RETRIES", "3"))
        base_delay = float(os.getenv("LLM_RETRY_BASE_DELAY", "5"))

        last_error = None
        for attempt in range(max_retries + 1):
            try:
                return self.client.chat.completions.create(**kwargs)
            except Exception as exc:
                last_error = exc
                if is_fatal_llm_error(exc):
                    raise FatalLLMError(f"Fatal LLM/API error during {purpose}: {exc}") from exc
                status_code = getattr(exc, "status_code", None)
                message = str(exc)
                transient = (
                    status_code in {408, 409, 429, 500, 502, 503, 504}
                    or "429" in message
                    or "rate" in message.lower()
                    or "timeout" in message.lower()
                    or "connection error" in message.lower()
                    or "connecterror" in message.lower()
                    or "nodename nor servname" in message.lower()
                    or "failed to lookup" in message.lower()
                    or "temporarily" in message.lower()
                    or "负载" in message
                    or "稍后再试" in message
                )
                if not transient or attempt >= max_retries:
                    raise

                delay = base_delay * (attempt + 1)
                print(f"         ⏳ Retrying LLM action call ({purpose}) in {delay:.1f}s ({attempt + 1}/{max_retries})...")
                time.sleep(delay)
        raise last_error

    def attach_event_logger(self, event_logger: Optional[JsonlEventLogger]):
        self.event_logger = event_logger
        self.command_recorder.attach_event_logger(event_logger)

    def _log_step(self, content):
        with open(self.realtime_log_path, "a", encoding="utf-8") as f:
            f.write(content + "\n")

    def _archive_action_script(self, script_path: str) -> Optional[str]:
        if not script_path or not os.path.exists(script_path):
            return None
        os.makedirs(self.action_archive_dir, exist_ok=True)
        base_name = os.path.basename(script_path)
        archive_path = os.path.join(self.action_archive_dir, base_name)
        if os.path.exists(archive_path):
            stem, ext = os.path.splitext(base_name)
            archive_path = os.path.join(self.action_archive_dir, f"{stem}_{int(time.time() * 1000)}{ext}")
        shutil.move(script_path, archive_path)
        if self.event_logger:
            self.event_logger.emit(
                "action_script_archived",
                {"from": script_path, "to": archive_path},
            )
        return archive_path

    def _rpc_host_port(self) -> Tuple[str, int]:
        parsed = urlparse(self.RPC_URL)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        return host, port

    def _replace_rpc_port(self, port: int):
        parsed = urlparse(self.RPC_URL)
        scheme = parsed.scheme or "http"
        host = parsed.hostname or "127.0.0.1"
        self.RPC_URL = f"{scheme}://{host}:{port}"

    def _pick_free_local_port(self) -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])

    def _rpc_is_reachable(self) -> bool:
        host, port = self._rpc_host_port()
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return True
        except OSError:
            return False

    def _rpc_json(self, method: str, params: Optional[List] = None, timeout: float = 3.0) -> Optional[Dict]:
        payload = json.dumps({
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params or [],
        }).encode("utf-8")
        request = urllib.request.Request(
            self.RPC_URL,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            if self.event_logger:
                self.event_logger.emit("rpc_call_failed", {
                    "method": method,
                    "error": str(exc),
                    "rpc_url": self.RPC_URL,
                })
            return None

    def _fund_actor_account(self, agent: Dict, wei_amount: Optional[int] = None) -> bool:
        address = agent.get("address")
        if not address:
            for data in self.accounts_config.values():
                if str(data.get("private_key", "")).lower() == str(agent.get("key", "")).lower():
                    address = data.get("address")
                    break
        if not address or not self._rpc_is_reachable():
            return False

        amount = wei_amount or int(os.getenv("LEVER_AGENT_FUND_WEI", str(10_000 * 10**18)))
        result = self._rpc_json("anvil_setBalance", [address, hex(amount)])
        ok = bool(result and "error" not in result)
        if self.event_logger:
            self.event_logger.emit("agent_account_funded", {
                "agent": agent.get("name"),
                "address": address,
                "wei": str(amount),
                "ok": ok,
                "error": (result or {}).get("error"),
            })
        return ok

    def _ensure_local_anvil(self):
        if os.getenv("LEVER_AUTO_ANVIL", "1").lower() in {"0", "false", "no"}:
            return

        host, port = self._rpc_host_port()
        if host not in {"127.0.0.1", "localhost", "::1"}:
            return

        if self._rpc_is_reachable():
            if self.event_logger:
                self.event_logger.emit("anvil_reused", {"rpc_url": self.RPC_URL})
            return

        anvil_bin = shutil.which("anvil")
        if not anvil_bin:
            raise RuntimeError("Local RPC is unreachable and `anvil` is not available on PATH.")

        if self._owned_anvil_process and self._owned_anvil_process.poll() is None:
            return

        if os.getenv("LEVER_AUTO_ANVIL_FREE_PORT", "1").lower() not in {"0", "false", "no"}:
            port = self._pick_free_local_port()
            self._replace_rpc_port(port)

        log_path = os.path.join(self.root_dir, "anvil.log")
        os.makedirs(self._owned_anvil_cache_dir, exist_ok=True)
        anvil_cmd = [
            anvil_bin,
            "-p",
            str(port),
            "--accounts",
            "20",
            "--balance",
            "10000",
            "--block-time",
            "1",
            "--prune-history",
            "1",
            "--cache-path",
            self._owned_anvil_cache_dir,
        ]
        self._owned_anvil_log = open(log_path, "w", encoding="utf-8")
        self._owned_anvil_process = subprocess.Popen(
            anvil_cmd,
            stdout=self._owned_anvil_log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        atexit.register(self._stop_owned_anvil)

        deadline = time.time() + 10
        while time.time() < deadline:
            if self._owned_anvil_process.poll() is not None:
                break
            if self._rpc_is_reachable():
                print(f"   ⛓️  Started local anvil at {self.RPC_URL}")
                if self.event_logger:
                    self.event_logger.emit(
                        "anvil_started",
                        {
                            "rpc_url": self.RPC_URL,
                            "log_path": log_path,
                            "cache_path": self._owned_anvil_cache_dir,
                            "disk_safe_flags": {
                                "prune_history": True,
                                "cache_path_is_run_local": True,
                            },
                        },
                    )
                return
            time.sleep(0.25)

        self._stop_owned_anvil()
        raise RuntimeError(f"Failed to start local anvil at {self.RPC_URL}; see {log_path}")

    def _cleanup_owned_anvil_cache(self):
        if os.getenv("LEVER_KEEP_ANVIL_CACHE", "0").lower() in {"1", "true", "yes"}:
            return
        cache_dir = getattr(self, "_owned_anvil_cache_dir", None)
        if not cache_dir:
            return
        shutil.rmtree(cache_dir, ignore_errors=True)
        if self.event_logger:
            self.event_logger.emit("anvil_cache_cleaned", {"cache_path": cache_dir})

    def _stop_owned_anvil(self):
        proc = self._owned_anvil_process
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
        self._owned_anvil_process = None
        if self._owned_anvil_log:
            self._owned_anvil_log.close()
            self._owned_anvil_log = None
        self._cleanup_owned_anvil_cache()

    def close(self):
        self._stop_owned_anvil()

    def _setup_fs(self):
        marker_path = os.path.join(self.root_dir, ".lever_workspace")
        if os.path.exists(self.root_dir):
            if not os.path.isfile(marker_path):
                raise RuntimeError(
                    f"Refusing to replace unmarked simulation directory: {self.root_dir}"
                )
            shutil.rmtree(self.root_dir)
        os.makedirs(self.src_dir, exist_ok=True)
        os.makedirs(self.script_dir, exist_ok=True)
        os.makedirs(self.action_archive_dir, exist_ok=True)
        os.makedirs(os.path.join(self.root_dir, "cache"), exist_ok=True)
        os.makedirs(os.path.join(self.root_dir, "broadcast"), exist_ok=True)
        with open(marker_path, "w", encoding="utf-8") as f:
            f.write("LeVer managed simulation workspace\n")

        shared_lib_path = os.path.abspath(os.getenv("SOLIDITY_LIB_PATH", os.path.abspath("libs_cache")))
        forge_std_src = os.path.join(shared_lib_path, "forge-std", "src")
        openzeppelin_root = os.path.join(shared_lib_path, "openzeppelin-contracts")

        if os.path.exists(forge_std_src) and os.path.exists(openzeppelin_root):
            remappings = [
                f"@openzeppelin/={openzeppelin_root}/",
                f"forge-std/={forge_std_src}/",
            ]
            libs_line = 'libs = []\n'
        else:
            print(f"   ⚠️ Shared Solidity libs not found at {shared_lib_path}; falling back to local lib remappings.")
            remappings = [
                "@openzeppelin/=lib/openzeppelin-contracts/",
                "forge-std/=lib/forge-std/src/",
            ]
            libs_line = 'libs = ["lib"]\n'

        with open(os.path.join(self.root_dir, "foundry.toml"), "w") as f:
            f.write('[profile.default]\n')
            f.write('src = "src"\n')
            f.write('out = "out"\n')
            f.write('cache_path = "cache"\n')
            f.write('optimizer = true\n')
            f.write('optimizer_runs = 200\n')
            f.write('via_ir = true\n')
            f.write(libs_line)
            f.write("remappings = [\n")
            for remapping in remappings:
                f.write(f'  "{remapping}",\n')
            f.write("]\n")

        with open(os.path.join(self.root_dir, "remappings.txt"), "w") as f:
            for remapping in remappings:
                f.write(remapping + "\n")

    def _remember_core_source(self, relative_path: str, content: str):
        rel = relative_path.replace("\\", "/").lstrip("/")
        if rel in {"src/Target.sol", "src/Mocks.sol", "src/SafetyRules.sol", "script/Deploy.s.sol", "script/CheckSafety.s.sol"}:
            self._core_source_snapshot[rel] = content

    def _ensure_core_sources_present(self, required: Optional[Tuple[str, ...]] = None) -> List[str]:
        required_paths = tuple(required or ("src/Target.sol", "src/Mocks.sol", "src/SafetyRules.sol"))
        restored: List[str] = []
        for rel in required_paths:
            rel = rel.replace("\\", "/").lstrip("/")
            content = self._core_source_snapshot.get(rel)
            if content is None:
                continue
            full_path = os.path.join(self.root_dir, rel)
            if os.path.exists(full_path):
                continue
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(content)
            restored.append(rel)
        if restored:
            print(f"      🧩 Restored simulation sources: {', '.join(restored)}")
            if self.event_logger:
                self.event_logger.emit("simulation_sources_self_healed", {"restored": restored})
        return restored

    def _prepare_foundry_script_dirs(self, script_name: str):
        base_name = os.path.basename(script_name)
        if not base_name:
            return
        chain_id = os.getenv("LEVER_FOUNDRY_CHAIN_ID", "31337")
        for root in ("cache", "broadcast"):
            os.makedirs(os.path.join(self.root_dir, root, base_name, chain_id), exist_ok=True)

    def _prepare_foundry_dirs_for_command(self, cmd: str):
        if "forge script" not in cmd:
            return
        for match in re.finditer(r"\bforge\s+script\s+script/([^\s]+\.s\.sol)\b", cmd):
            self._prepare_foundry_script_dirs(match.group(1))

    def _command_timed_out(self, stderr: str) -> bool:
        return "[TIMEOUT]" in (stderr or "")

    def _safety_check_timeout(self) -> float:
        return float(os.getenv("LEVER_SAFETY_CHECK_TIMEOUT", "120"))

    def _run_command(self, cmd, *, label: Optional[str] = None, kind: str = "foundry", timeout_s: Optional[float] = None):
        env = os.environ.copy()
        if "FOUNDRY_REMAPPINGS" in env:
            del env["FOUNDRY_REMAPPINGS"]

        if self.ANVIL_KEYS:
            env["PRIVATE_KEY"] = self.ANVIL_KEYS[0]

        print(f"\n⚡ [CMD]: {cmd}")

        self._prepare_foundry_dirs_for_command(cmd)
        timeout_s = float(timeout_s if timeout_s is not None else os.getenv("LEVER_COMMAND_TIMEOUT", "600"))
        started_at = time.time()
        run_cmd = shlex.split(cmd)
        try:
            result = subprocess.run(
                run_cmd,
                shell=False,
                cwd=self.root_dir,
                capture_output=True,
                text=True,
                env=env,
                timeout=timeout_s,
            )
            stdout = result.stdout
            stderr = result.stderr
            returncode = result.returncode
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout or ""
            stderr = exc.stderr or ""
            if isinstance(stdout, bytes):
                stdout = stdout.decode(errors="replace")
            if isinstance(stderr, bytes):
                stderr = stderr.decode(errors="replace")
            stderr = (
                f"{stderr}\n[TIMEOUT] Command exceeded {timeout_s:.0f}s and was terminated."
            ).strip()
            returncode = 124
        ended_at = time.time()
        command_record = self.command_recorder.record(
            cmd=cmd,
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
            cwd=self.root_dir,
            started_at=started_at,
            ended_at=ended_at,
            label=label,
            kind=kind,
        )
        self.command_history.append(command_record)

        if stdout.strip():
            stdout_preview = text_tail(stdout.strip(), 2500)
            print(f"----- STDOUT ({command_record['label']}) -----\n{stdout_preview}\n------------------")

        if stderr.strip():
            if returncode != 0:
                stderr_preview = text_tail(stderr.strip(), 2500)
                print(f"----- STDERR (FAILED: {command_record['label']}) -----\n{stderr_preview}\n------------------")
            else:
                # Filter compiler warnings.
                filtered_lines = []
                skip_block = False
                for line in stderr.split('\n'):
                    if "Warning" in line or "note[" in line or "note:" in line:
                        skip_block = True
                        continue
                    if skip_block and line.strip() and not line.strip().startswith(('-->', '|', '^', 'help:', '=')):
                        if "Error" in line or "Compiling" in line:
                            skip_block = False
                        else:
                            pass
                    if not skip_block:
                        filtered_lines.append(line)

                clean_stderr = "\n".join(filtered_lines).strip()
                if clean_stderr:
                    print(f"----- STDERR (CLEANED: {command_record['label']}) -----\n{text_tail(clean_stderr, 2500)}\n------------------")

        return stdout, stderr, returncode

    def _describe_action_outcome(self, output_log: str) -> Tuple[str, str]:
        if output_log.startswith("[ACTION_CONFIRMED_ONCHAIN"):
            return "confirmed", "✅ Action confirmed on-chain."
        if output_log.startswith("[ACTION_PARTIAL_EFFECT_OBSERVED]"):
            return "partial_effect_observed", "🟡 Action ended unconfirmed, but semantic state-changing evidence was observed."
        if output_log.startswith("[ACTION_SIMULATED_BROADCAST_FAILED]"):
            return "simulated_broadcast_failed", "⚠️ Action simulated successfully, but broadcast failed."
        if output_log.startswith("[ACTION_SIMULATION_FAILED_OR_REVERTED]"):
            return "simulated_reverted", "🛡️ Action simulation reverted or failed; no confirmed on-chain state change."
        if output_log.startswith("[ATTACK REVERTED / DEFENSE SUCCESS]"):
            return "simulated_reverted", "🛡️ Action reverted or was blocked."
        return "unknown", "✅ Action completed."

    def _has_semantic_effect_evidence(self, output: str) -> bool:
        """Detect useful state-changing evidence hidden inside an unconfirmed script."""
        if not output:
            return False
        success_patterns = [
            r'console::log\(".*(?:mint|deposit|withdraw|claim|redeem|distribute|settle|buy|purchase|airdrop|sendFunds|create|clone|deploy|setState|setPrice|setWhitelist|transferOwnership|setVault|setMinter).*?(?:success|succeeded| ok|_ok|Success:).*"\)',
            r'console::log\(".*(?:success|succeeded| ok|_ok|Success:).*?(?:mint|deposit|withdraw|claim|redeem|distribute|settle|buy|purchase|airdrop|sendFunds|create|clone|deploy|setState|setPrice|setWhitelist|transferOwnership|setVault|setMinter).*"\)',
            r'\b(?:mint|deposit|withdraw|claim|redeem|distribute|settle|buy|purchase|create|clone|deploy|airdrop|sendFunds|setState|setPrice|setWhitelist|transferOwnership|setVault|setMinter)\w*\([^)]*\)\s*\n\s*│\s+├─\s+emit',
            r'emit\s+(?:Transfer|Minted|Deposit|Withdraw|Claim|Redeem|Swap|FundsDistributed|TeamWithdrawal|OwnershipTransferred|Approval)\b',
        ]
        return any(re.search(pattern, output, flags=re.IGNORECASE) for pattern in success_patterns)

    # Multi-round agent simulation with per-agent memory and a compact actor set.
    def _attack_result(
        self,
        safe: bool,
        log: str,
        *,
        attack_episodes: int = 0,
        breaches: int = 0,
        traces: Optional[List[str]] = None,
        infra_broken: bool = False,
        deployment_success: bool = False,
        address_parse_success: bool = False,
        agent_steps: int = 0,
        malicious_steps: int = 0,
        total_rounds: Optional[int] = None,
        schedule_phases: Optional[List[Dict]] = None,
        honest_intents: Optional[List[Dict]] = None,
        front_run_attempts: int = 0,
        back_run_attempts: int = 0,
        lifecycle_setup: Optional[Dict] = None,
        foundry_summary: Optional[Dict] = None,
    ) -> Dict:
        return {
            "safe": safe,
            "log": log,
            "attack_episodes": attack_episodes,
            "breaches": breaches,
            "attack_success_rate": breaches / attack_episodes if attack_episodes else 0.0,
            "traces": traces or [],
            "infra_broken": infra_broken,
            "deployment_success": deployment_success,
            "address_parse_success": address_parse_success,
            "agent_steps": agent_steps,
            "malicious_steps": malicious_steps,
            "total_rounds": total_rounds if total_rounds is not None else self.attack_rounds,
            "scheduler": "pomp_interleaving",
            "schedule_phases": schedule_phases or [],
            "honest_intents": honest_intents or [],
            "front_run_attempts": front_run_attempts,
            "back_run_attempts": back_run_attempts,
            "lifecycle_setup": lifecycle_setup or {},
            "foundry_summary": foundry_summary or {},
        }

    def _short_intent(self, action_body: str) -> str:
        action = action_body or ""
        if "/* [ACTION] */" in action:
            action = action.split("/* [ACTION] */", 1)[1]
        action = re.sub(r"/\* \[HELPER\] \*/[\s\S]*", "", action)
        action = re.sub(r"\s+", " ", action).strip()
        return action[:1200]

    def _build_phase_context(self, phase: str, pending_intent: Optional[Dict] = None) -> str:
        lines = [f"[SCHEDULER PHASE] {phase}"]
        if pending_intent:
            lines.extend([
                "[PENDING HONEST INTENT]",
                f"Honest agent: {pending_intent.get('agent')}",
                f"Planned action: {pending_intent.get('intent')}",
            ])
        return "\n".join(lines)

    def _build_lifecycle_setup_context(self, addresses: Dict) -> str:
        agent_lines = []
        for name, data in self.accounts_config.items():
            if name == "Deployer":
                continue
            agent_lines.append(f"- {name} ({data['role']}): {data['address']}")

        return f"""
        [SCHEDULER PHASE] lifecycle_setup
        [PURPOSE]
        Prepare a valid business lifecycle state before adversarial scheduling starts.
        This is not an attack and not a safety verdict. The goal is to avoid judging
        mint/claim/distribute/deposit/proxy behavior while the contract is still in an
        unopened or uninitialized phase.

        [SETUP ACTOR]
        You are a lifecycle setup operator using one configured local Anvil account.
        Many generated contracts make the deployer the owner/admin, but some deploy
        scripts assign owner/minter/admin roles to Alice, Bob, or another configured
        account. Use only exposed calls that the current actor is authorized to make.
        Do not use prank cheatcodes and do not impersonate arbitrary owners.

        [KNOWN PARTICIPANTS]
        {chr(10).join(agent_lines)}

        [SETUP OBJECTIVES - GENERIC]
        - PRIORITY 1 is lifecycle reachability. If the interface exposes a state,
          phase, sale, presale, public sale, distribution, initialization, or start
          function, call the minimal authorized setup that enables normal user
          behavior before doing anything else. For enum-based state setters, prefer
          the state whose name means Active/Open/Started/Distribution/Live when it is
          present in the imported target types.
        - If the target exposes init/initialize/start/open/activate/state-transition
          functions, call the minimal authorized sequence that reaches the main usable
          phase described by the interface and requirement.
        - If the target exposes sale/presale/public-sale switches, open the phase most
          relevant to user mint/buy behavior. For mutually exclusive lifecycle modes
          (for example presale vs public sale, commit vs reveal, active vs distribution),
          choose exactly one useful phase and stop there. Do not open two mutually
          exclusive phases in the same setup script.
        - If the target exposes whitelist/allowlist functions, include the honest users
          and, when the semantic objective concerns adversarial repeated minting, include
          at least one malicious actor too. This makes later cap checks meaningful.
        - If the target exposes price/fee configuration, set a small nonzero value unless
          the constructor already did so.
        - Do not invent or change security thresholds such as caps, quotas, allocation
          limits, reserves, timelocks, or per-user limits unless the contract cannot reach
          any positive path without that configuration. Those parameters are part of what
          later agents and checkers should evaluate, not lifecycle conveniences.
        - Do not perform user-facing or settlement actions in lifecycle setup:
          no mint/buy/deposit/withdraw/claim/redeem/distribute/sendFunds stress calls.
          Those calls are intentionally left for honest and malicious agents after setup.
          Exception: for token-like contracts that expose transfer(address,uint*) and
          balanceOf(address), a small transfer of existing deployer-held tokens to the
          known participants is allowed so later transfer/approval/callback paths are
          reachable. Do not mint new supply solely for this setup convenience.
        - If the target exposes factory/proxy/clone creation, create one ordinary instance
          and log its address/code length if possible.
        - Keep setup small and reversible-looking: no draining, no exploit probes, no
          repeated stress calls. Repeated/stress behavior belongs to the later agents.
        - Do NOT intentionally execute calls that are expected to revert just to probe
          security. In setup, a known failing call often prevents useful state from
          being broadcast. Read-only probes are fine; mutating calls should be ones you
          expect to succeed.
        - try/catch is for genuinely uncertain setup calls. It is not permission to include
          mutating fallback calls that are likely to revert because of a phase mutex or an
          already-open mode.
        - If a setup call fails unexpectedly, catch it, log a fixed tag, and continue
          with other independent setup actions. End the script after successful or
          neutral setup work, not after an avoidable revert probe.
        - A good setup script usually contains only calls like setState/openSale,
          setPresaleState, setPublicSaleState, initialize/init/start, setWhitelist,
          setPrice/setFee, setOwner metadata, or one factory create/clone call. It should
          avoid calls that move user assets or test authorization.

        [ADDRESSES]
        {json.dumps(addresses)}
        """

    def _infer_target_contract_name(self, interface_desc: str) -> str:
        fallback = "Target"
        base_like = {
            "Ownable",
            "ERC20",
            "ERC721",
            "ERC721Core",
            "ERC1155",
            "ReentrancyGuard",
            "Pausable",
            "AccessControl",
        }
        for match in re.finditer(r"Contract Name:\s*([A-Za-z_][A-Za-z0-9_]*)", interface_desc or ""):
            name = match.group(1)
            if not name.startswith("I") and not name.lower().startswith("mock") and name not in base_like:
                return name
            if fallback == "Target":
                fallback = name
        return fallback

    def _build_deterministic_lifecycle_action(self, addresses: Dict, interface_desc: str) -> str:
        target_addr = addresses.get("TARGET")
        contract_name = self._primary_target_contract_name() or self._infer_target_contract_name(interface_desc)
        iface = interface_desc or ""
        target_source = ""
        try:
            with open(os.path.join(self.src_dir, "Target.sol"), "r", encoding="utf-8") as f:
                target_source = f.read()
        except OSError:
            target_source = ""
        source_context = f"{iface}\n{target_source}"
        participant_addresses = [
            data["address"]
            for name, data in self.accounts_config.items()
            if name != "Deployer" and data.get("address")
        ]

        lines = [
            "/* [HELPER] */",
            "// Deterministic lifecycle setup fallback; no helper contracts needed.",
            "",
            "/* [ACTION] */",
            f"address targetAddress = {target_addr};",
            f"{contract_name} target = {contract_name}(targetAddress);",
            'console.log("Deterministic lifecycle setup");',
            "console.logAddress(targetAddress);",
            f"address setupDeployer = {self.accounts_config.get('Deployer', {}).get('address', '0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266')};",
            "bool setupHasKnownController = false;",
            "bool setupActorCanControl = false;",
            "{",
            '    (bool okOwner, bytes memory dataOwner) = targetAddress.staticcall(abi.encodeWithSignature("owner()"));',
            "    if (okOwner && dataOwner.length >= 32) {",
            "        address ownerAddr = abi.decode(dataOwner, (address));",
            "        setupHasKnownController = setupHasKnownController || ownerAddr != address(0);",
            "        setupActorCanControl = setupActorCanControl || actorEOA == ownerAddr;",
            '        console.log("setup owner");',
            "        console.logAddress(ownerAddr);",
            "    }",
            '    (bool okAdmin, bytes memory dataAdmin) = targetAddress.staticcall(abi.encodeWithSignature("admin()"));',
            "    if (okAdmin && dataAdmin.length >= 32) {",
            "        address adminAddr = abi.decode(dataAdmin, (address));",
            "        setupHasKnownController = setupHasKnownController || adminAddr != address(0);",
            "        setupActorCanControl = setupActorCanControl || actorEOA == adminAddr;",
            '        console.log("setup admin");',
            "        console.logAddress(adminAddr);",
            "    }",
            '    (bool okMinter, bytes memory dataMinter) = targetAddress.staticcall(abi.encodeWithSignature("minter()"));',
            "    if (okMinter && dataMinter.length >= 32) {",
            "        address minterAddr = abi.decode(dataMinter, (address));",
            '        console.log("setup minter");',
            "        console.logAddress(minterAddr);",
            "    }",
            '    (bool okController, bytes memory dataController) = targetAddress.staticcall(abi.encodeWithSignature("controller()"));',
            "    if (okController && dataController.length >= 32) {",
            "        address controllerAddr = abi.decode(dataController, (address));",
            "        setupHasKnownController = setupHasKnownController || controllerAddr != address(0);",
            "        setupActorCanControl = setupActorCanControl || actorEOA == controllerAddr;",
            '        console.log("setup controller");',
            "        console.logAddress(controllerAddr);",
            "    }",
            '    (bool okManager, bytes memory dataManager) = targetAddress.staticcall(abi.encodeWithSignature("manager()"));',
            "    if (okManager && dataManager.length >= 32) {",
            "        address managerAddr = abi.decode(dataManager, (address));",
            "        setupHasKnownController = setupHasKnownController || managerAddr != address(0);",
            "        setupActorCanControl = setupActorCanControl || actorEOA == managerAddr;",
            '        console.log("setup manager");',
            "        console.logAddress(managerAddr);",
            "    }",
            '    (bool okOperator, bytes memory dataOperator) = targetAddress.staticcall(abi.encodeWithSignature("operator()"));',
            "    if (okOperator && dataOperator.length >= 32) {",
            "        address operatorAddr = abi.decode(dataOperator, (address));",
            "        setupHasKnownController = setupHasKnownController || operatorAddr != address(0);",
            "        setupActorCanControl = setupActorCanControl || actorEOA == operatorAddr;",
            '        console.log("setup operator");',
            "        console.logAddress(operatorAddr);",
            "    }",
            "}",
            "bool setupCanAttemptAdmin = setupActorCanControl || (!setupHasKnownController && actorEOA == setupDeployer);",
            "if (!setupCanAttemptAdmin) {",
            '    console.log("setup actor is not known controller; admin setup calls will be skipped");',
            "}",
        ]

        def func_params(name: str) -> List[str]:
            patterns = [
                rf"\bfunction\s+{re.escape(name)}\s*\(([^)]*)\)",
                rf"(?:^|\n)\s*-\s*{re.escape(name)}\s*\(([^)]*)\)",
                rf"(?:^|\n)\s*-\s*function\s+{re.escape(name)}\s*\(([^)]*)\)",
            ]
            params: List[str] = []
            for pattern in patterns:
                params.extend(match.group(1) for match in re.finditer(pattern, source_context))
            return params

        def has_params(name: str, *needles: str) -> bool:
            return any(all(re.search(needle, params) for needle in needles) for params in func_params(name))

        def has_no_arg(name: str) -> bool:
            return any(params.strip() == "" for params in func_params(name))

        def has_address_uint_params(name: str) -> bool:
            return any(
                re.search(r"\baddress\b(?!\s*\[)", params) and re.search(r"\buint(?:8|16|32|64|128|256)?\b", params)
                for params in func_params(name)
            )

        def uint_only_param_types(params: str, count: int) -> List[str]:
            parts = [part.strip() for part in params.split(",") if part.strip()]
            if len(parts) != count:
                return []
            types: List[str] = []
            for part in parts:
                match = re.search(r"\buint(8|16|32|64|128|256)?\b", part)
                if not match:
                    return []
                types.append(f"uint{match.group(1) or '256'}")
            return types

        def cast_uint_expr(uint_type: str, expr: str) -> str:
            return f"{uint_type}({expr})"

        def add_direct_try(call_expr: str, label: str, *, admin_guard: bool = True):
            if admin_guard:
                lines.extend([
                    "if (setupCanAttemptAdmin) {",
                    f"    try target.{call_expr} {{",
                    f'        console.log("setup {label} success");',
                    "    } catch {",
                    f'        console.log("setup {label} skipped");',
                    "    }",
                    "} else {",
                    f'    console.log("setup {label} skipped: actor not controller");',
                    "}",
                ])
                return
            lines.extend([
                f"try target.{call_expr} {{",
                f'    console.log("setup {label} success");',
                "} catch {",
                f'    console.log("setup {label} skipped");',
                "}",
            ])

        def add_low_level_call(signature: str, args_expr: str, label: str, *, admin_guard: bool = True):
            suffix = f", {args_expr}" if args_expr else ""
            if admin_guard:
                lines.extend([
                    "if (setupCanAttemptAdmin) {",
                    "{",
                    f'    (bool ok, ) = targetAddress.call(abi.encodeWithSignature("{signature}"{suffix}));',
                    "    if (ok) {",
                    f'        console.log("setup {label} success");',
                    "    } else {",
                    f'        console.log("setup {label} skipped");',
                    "    }",
                    "}",
                    "} else {",
                    f'    console.log("setup {label} skipped: actor not controller");',
                    "}",
                ])
                return
            lines.extend([
                "{",
                f'    (bool ok, ) = targetAddress.call(abi.encodeWithSignature("{signature}"{suffix}));',
                "    if (ok) {",
                f'        console.log("setup {label} success");',
                "    } else {",
                f'        console.log("setup {label} skipped");',
                "    }",
                "}",
            ])

        def add_low_level_calls_for_signatures(signatures: List[Tuple[str, str, str]]):
            for signature, args_expr, label in signatures:
                if re.search(rf"\b{re.escape(signature.split('(')[0])}\s*\(", iface):
                    add_low_level_call(signature, args_expr, label)

        def add_state_candidate_sequence(name: str, values: List[int]):
            if not re.search(rf"\b{re.escape(name)}\s*\(", source_context):
                return
            signature_args: List[Tuple[str, str]] = []
            if has_params(name, r"\buint(?:256)?\b"):
                signature_args.append((f"{name}(uint256)", "uint256"))
            if has_params(name, r"\buint8\b") or has_params(name, r"\bContractState\b") or not signature_args:
                signature_args.insert(0, (f"{name}(uint8)", "uint8"))
            seen = set()
            signature_args = [item for item in signature_args if not (item[0] in seen or seen.add(item[0]))]
            flag_name = f"setup{name[0].upper()}{name[1:]}Done"
            lines.extend([
                "if (setupCanAttemptAdmin) {",
                f"    bool {flag_name} = false;",
            ])
            for signature, arg_type in signature_args:
                for value in values:
                    value_expr = f"{arg_type}({value})"
                    lines.extend([
                        f"    if (!{flag_name}) {{",
                        f'        (bool ok, ) = targetAddress.call(abi.encodeWithSignature("{signature}", {value_expr}));',
                        "        if (ok) {",
                        f"            {flag_name} = true;",
                        f'            console.log("setup {name} {value_expr} success");',
                        "        } else {",
                        f'            console.log("setup {name} {value_expr} skipped");',
                        "        }",
                        "    }",
                    ])
            lines.extend([
                "} else {",
                f'    console.log("setup {name} skipped: actor not controller");',
                "}",
            ])

        def add_parameterized_lifecycle_call(name: str):
            for params in func_params(name):
                uint_types = uint_only_param_types(params, 2)
                if not uint_types:
                    continue
                lower_name = name.lower()
                lower_params = params.lower()
                signature = f"{name}({uint_types[0]},{uint_types[1]})"

                if re.search(r"\b(price|cost|fee|rate)\b", lower_params):
                    args_expr = ", ".join([
                        cast_uint_expr(uint_types[0], "0.01 ether"),
                        cast_uint_expr(uint_types[1], "7 days"),
                    ])
                    label = f"{name} price-duration"
                elif "block" in lower_params or "fund" in lower_name:
                    args_expr = ", ".join([
                        cast_uint_expr(uint_types[0], "block.number"),
                        cast_uint_expr(uint_types[1], "block.number + 256"),
                    ])
                    label = f"{name} block-window"
                elif re.search(r"\b(start|begin|end|stop|time|timestamp)\b", lower_params):
                    args_expr = ", ".join([
                        cast_uint_expr(uint_types[0], "block.timestamp"),
                        cast_uint_expr(uint_types[1], "block.timestamp + 7 days"),
                    ])
                    label = f"{name} time-window"
                elif "sale" in lower_name:
                    args_expr = ", ".join([
                        cast_uint_expr(uint_types[0], "0.01 ether"),
                        cast_uint_expr(uint_types[1], "7 days"),
                    ])
                    label = f"{name} sale-defaults"
                else:
                    args_expr = ", ".join([
                        cast_uint_expr(uint_types[0], "block.number"),
                        cast_uint_expr(uint_types[1], "block.number + 256"),
                    ])
                    label = f"{name} uint-window"

                add_low_level_call(signature, args_expr, label)
                return

        for name in ["setPrice", "setMintPrice", "setFee"]:
            if has_params(name, r"\buint"):
                add_direct_try(f"{name}(0.01 ether)", name)

        if func_params("setWhitelist"):
            lines.append(f"address[] memory wl = new address[]({len(participant_addresses)});")
            for idx, addr in enumerate(participant_addresses):
                lines.append(f"wl[{idx}] = {addr};")
            if has_params("setWhitelist", r"address\s*\[\s*\]", r"\bbool\b"):
                add_direct_try("setWhitelist(wl, true)", "setWhitelist array")
            elif has_params("setWhitelist", r"\baddress\b(?!\s*\[)", r"\bbool\b"):
                for idx, addr in enumerate(participant_addresses):
                    add_direct_try(f"setWhitelist({addr}, true)", f"setWhitelist single {idx}")
            else:
                add_low_level_call("setWhitelist(address[],bool)", "wl, true", "setWhitelist array fallback")

        if func_params("setAllowlist") or func_params("setAllowList"):
            lines.append(f"address[] memory al = new address[]({len(participant_addresses)});")
            for idx, addr in enumerate(participant_addresses):
                lines.append(f"al[{idx}] = {addr};")
            for name in ["setAllowlist", "setAllowList"]:
                if has_params(name, r"address\s*\[\s*\]", r"\bbool\b"):
                    add_direct_try(f"{name}(al, true)", f"{name} array")
                elif has_params(name, r"\baddress\b(?!\s*\[)", r"\bbool\b"):
                    for idx, addr in enumerate(participant_addresses):
                        add_direct_try(f"{name}({addr}, true)", f"{name} single {idx}")

        for name in ["addToWhitelist", "addWhitelist", "whitelistAddress", "allowlistAddress", "addToAllowlist", "addToAllowList"]:
            if has_params(name, r"\baddress\b(?!\s*\[)"):
                add_direct_try(f"{name}(actorEOA)", f"{name} actor")
                for idx, addr in enumerate(participant_addresses):
                    add_direct_try(f"{name}({addr})", f"{name} {idx}")

        if re.search(r"\bsetPresaleState\s*\(", iface):
            if has_params("setPresaleState", r"\bbool\b"):
                add_direct_try("setPresaleState(true)", "setPresaleState")
            else:
                add_low_level_call("setPresaleState(bool)", "true", "setPresaleState fallback")
        elif re.search(r"\bsetPublicSaleState\s*\(", iface):
            if has_params("setPublicSaleState", r"\bbool\b"):
                add_direct_try("setPublicSaleState(true)", "setPublicSaleState")
            else:
                add_low_level_call("setPublicSaleState(bool)", "true", "setPublicSaleState fallback")

        enum_sale_match = re.search(
            r"\b(interface|contract)\s+([A-Za-z_][A-Za-z0-9_]*)[\s\S]{0,2500}?\benum\s+SaleState\s*\{([^}]*)\}",
            target_source,
        )
        if func_params("setSaleState") and enum_sale_match and has_params("setSaleState", r"\bSaleState\b"):
            enum_owner = enum_sale_match.group(2)
            values = [value.strip().split()[0] for value in enum_sale_match.group(3).split(",") if value.strip()]
            priority = ["Ongoing", "Open", "Public", "PublicSale", "Active", "Started", "Live", "Presale", "Sale", "Distribution"]
            chosen = next((name for name in priority if name in values), values[1] if len(values) > 1 else values[0] if values else None)
            if chosen:
                add_direct_try(f"setSaleState({enum_owner}.SaleState.{chosen})", f"setSaleState {chosen}")
        else:
            if has_params("setSaleState", r"\buint8\b"):
                add_direct_try("setSaleState(uint8(1))", "setSaleState uint8")
            elif has_params("setSaleState", r"\buint(?:256)?\b"):
                add_direct_try("setSaleState(uint256(1))", "setSaleState uint256")
            elif re.search(r"\bsetSaleState\s*\(", iface):
                add_low_level_call("setSaleState(uint8)", "uint8(1)", "setSaleState uint8 fallback")

        enum_state_match = re.search(
            r"\b(interface|contract)\s+([A-Za-z_][A-Za-z0-9_]*)[\s\S]{0,2500}?\benum\s+ContractState\s*\{([^}]*)\}",
            target_source,
        )
        if func_params("setState") and enum_state_match and has_params("setState", r"\bContractState\b"):
            enum_owner = enum_state_match.group(2)
            values = [value.strip().split()[0] for value in enum_state_match.group(3).split(",") if value.strip()]
            priority = ["Active", "Open", "Ongoing", "Started", "Live", "Public", "PublicSale", "Presale", "Sale", "Distribution"]
            chosen = next((name for name in priority if name in values), values[1] if len(values) > 1 else values[0] if values else None)
            if chosen:
                add_direct_try(f"setState({enum_owner}.ContractState.{chosen})", f"setState {chosen}")
        elif has_params("setState", r"\buint8\b"):
            add_direct_try("setState(uint8(1))", "setState uint8")
        elif has_params("setState", r"\buint(?:256)?\b"):
            add_direct_try("setState(uint256(1))", "setState uint256")
        elif re.search(r"\bsetState\s*\(", iface):
            add_low_level_call("setState(uint8)", "uint8(1)", "setState uint8 fallback")

        # Contracts often encode lifecycle gates as uint/enum-style state numbers
        # under names more specific than setState. Try active-like nonzero states
        # before token seeding so later positive-path actions can be reached.
        for name in ["setContractState", "setCurrentState"]:
            add_state_candidate_sequence(name, [2, 1])

        for name in ["setSaleActive", "setMintingActive", "setPublicSaleActive", "setPresaleActive"]:
            if has_params(name, r"\bbool\b"):
                add_direct_try(f"{name}(true)", name)
        if has_params("setPaused", r"\bbool\b"):
            add_direct_try("setPaused(false)", "setPaused false")
        for name in [
            "startFunding",
            "beginFunding",
            "startCrowdsale",
            "startICO",
            "startSale",
            "beginSale",
            "startPublicSale",
            "openPublicSale",
            "startPresale",
            "openPresale",
            "openSale",
        ]:
            add_parameterized_lifecycle_call(name)
        for name in ["unpause", "openSale", "startSale", "start", "activate", "open", "initialize", "init"]:
            if has_no_arg(name):
                add_direct_try(f"{name}()", name)

        if (
            participant_addresses
            and has_address_uint_params("transfer")
            and has_params("balanceOf", r"\baddress\b(?!\s*\[)")
        ):
            needed_seed = max(1, len(participant_addresses)) * 1000
            lines.extend([
                "{",
                "    uint256 setupTokenSeed = 1000;",
                f"    uint256 setupTokenSeedNeeded = {needed_seed};",
                "    (bool setupBalOk, bytes memory setupBalData) = targetAddress.staticcall(abi.encodeWithSignature(\"balanceOf(address)\", actorEOA));",
                "    if (setupBalOk && setupBalData.length >= 32 && abi.decode(setupBalData, (uint256)) >= setupTokenSeedNeeded) {",
            ])
            for idx, addr in enumerate(participant_addresses):
                lines.extend([
                    f'        (bool setupTransferOk{idx}, bytes memory setupTransferData{idx}) = targetAddress.call(abi.encodeWithSignature("transfer(address,uint256)", {addr}, setupTokenSeed));',
                    f"        if (setupTransferOk{idx} && (setupTransferData{idx}.length == 0 || (setupTransferData{idx}.length >= 32 && abi.decode(setupTransferData{idx}, (bool))))) {{",
                    f'            console.log("setup seedToken {idx} success");',
                    "        } else {",
                    f'            console.log("setup seedToken {idx} skipped");',
                    "        }",
                ])
            lines.extend([
                "    } else {",
                '        console.log("setup seedToken skipped");',
                "    }",
                "}",
            ])

        if func_params("addValidJob") or func_params("setValidJob"):
            job_addresses = []
            for tag, addr in addresses.items():
                if tag != "TARGET" and addr and addr != "0x0000000000000000000000000000000000000000":
                    job_addresses.append(addr)
            for idx, addr in enumerate(job_addresses[:6]):
                if has_params("addValidJob", r"\baddress\b(?!\s*\[)"):
                    add_direct_try(f"addValidJob({addr})", f"addValidJob {idx}")
                if has_params("setValidJob", r"\baddress\b(?!\s*\[)", r"\bbool\b"):
                    add_direct_try(f"setValidJob({addr}, true)", f"setValidJob {idx}")

        if re.search(r"\bcurrentState\s*\(", iface):
            lines.extend([
                '(bool currentStateOk, bytes memory currentStateData) = targetAddress.staticcall(abi.encodeWithSignature("currentState()"));',
                "if (currentStateOk && currentStateData.length >= 32) {",
                '    console.log("setup currentState");',
                "    console.logUint(abi.decode(currentStateData, (uint256)));",
                "} else {",
                '    console.log("setup currentState read skipped");',
                "}",
            ])
        if re.search(r"\bpresaleOpen\s*\(", iface):
            lines.extend([
                "try target.presaleOpen() returns (bool open) {",
                '    console.log("setup presaleOpen");',
                "    console.logUint(open ? 1 : 0);",
                "} catch {}",
            ])
        if re.search(r"\bpublicSaleOpen\s*\(", iface):
            lines.extend([
                "try target.publicSaleOpen() returns (bool open) {",
                '    console.log("setup publicSaleOpen");',
                "    console.logUint(open ? 1 : 0);",
                "} catch {}",
            ])

        lines.append("")
        return "\n".join(lines)

    def _run_lifecycle_setup(self, addresses: Dict, interface_desc: str) -> Dict:
        if os.getenv("LEVER_LIFECYCLE_SETUP", "1").lower() in {"0", "false", "no"}:
            return {"enabled": False, "status": "skipped"}

        print("   🧭 Running lifecycle setup before agent scheduling...")
        deterministic_action = self._build_deterministic_lifecycle_action(addresses, interface_desc)
        use_deterministic = os.getenv("LEVER_DETERMINISTIC_LIFECYCLE", "1").lower() not in {"0", "false", "no"}

        account_items = list(self.accounts_config.items()) if use_deterministic else [("Deployer", self.accounts_config.get("Deployer", {}))]
        if not any(data.get("private_key") for _, data in account_items):
            return {"enabled": True, "status": "missing_setup_key"}

        results = []
        any_success = False
        any_confirmed = False
        action_body = deterministic_action
        memory = ""
        for account_name, account in account_items:
            if not account.get("private_key"):
                continue
            setup_agent = {
                "name": f"LifecycleSetup_{account_name}",
                "role": "LIFECYCLE_SETUP_OPERATOR",
                "key": account["private_key"],
                "address": account.get("address"),
            }
            success, output_log, this_action_body, this_memory = self._run_agent_with_retry(
                setup_agent,
                0,
                addresses,
                interface_desc,
                history=[],
                allow_skip=False,
                previous_memory="",
                phase_context=self._build_lifecycle_setup_context(addresses),
                initial_action_body=deterministic_action if use_deterministic else None,
            )
            status, message = self._describe_action_outcome(output_log) if success else ("failed", "❌ Lifecycle setup failed.")
            print(f"      [{account_name}] {message}")
            results.append({
                "account": account_name,
                "address": account.get("address"),
                "success": success,
                "status": status,
                "log_tail": output_log[-1200:] if output_log else "",
            })
            any_success = any_success or success
            any_confirmed = any_confirmed or status == "confirmed"
            action_body = this_action_body or action_body
            memory = this_memory or memory
            if not use_deterministic:
                break

        aggregate_status = "confirmed" if any_confirmed else ("no_confirmed_setup" if any_success else "failed")
        message = "✅ Lifecycle setup attempts completed." if any_success else "❌ Lifecycle setup failed."
        print(f"      {message}")
        return {
            "enabled": True,
            "success": any_success,
            "status": aggregate_status,
            "action": self._short_intent(action_body),
            "memory": memory,
            "log_tail": "\n\n".join(f"[{r['account']}][{r['status']}]\n{r['log_tail']}" for r in results)[-3000:],
            "attempts": results,
        }

    def run_fuzz_and_attack(self, target_sol: str, mocks_sol: str, safety_sol: str, deploy_script: str, regex_config: Dict, interface_desc: str, safety_check_logic: str) -> Dict:
        self._setup_fs()
        try:
            self._ensure_local_anvil()
        except Exception as exc:
            return self._attack_result(
                False,
                f"Local anvil/RPC startup failed: {exc}",
                infra_broken=True,
                deployment_success=False,
                foundry_summary={"anvil": {"success": False, "error": str(exc), "rpc_url": self.RPC_URL}},
            )

        # 1. Write Files
        self.save_file("src/Target.sol", target_sol)
        self.save_file("src/Mocks.sol", mocks_sol)
        self.save_file("src/SafetyRules.sol", safety_sol)
        deploy_script = self._sanitize_deploy_script(deploy_script)
        self.save_file("script/Deploy.s.sol", deploy_script)

        # 2. Deploy
        print("   🌍 Deploying Simulation Environment...")
        self._ensure_core_sources_present(("src/Target.sol", "src/Mocks.sol", "src/SafetyRules.sol", "script/Deploy.s.sol"))
        deployer_key = self.ANVIL_KEYS[0]
        cmd = f"forge script script/Deploy.s.sol --broadcast --rpc-url {self.RPC_URL} --private-key {deployer_key}"
        out, err, code = self._run_command(cmd, label="agent_deploy", kind="foundry_deploy")
        deploy_summary = parse_foundry_output(out, err, code)

        if code != 0:
            return self._attack_result(
                False,
                f"Deployment Failed:\n{err}",
                infra_broken=True,
                deployment_success=False,
                foundry_summary={"deploy": deploy_summary},
            )

        # 3. Parse Addresses
        addresses = dict(deploy_summary.get("deployed_tags", {}))
        for label, pattern in regex_config.items():
            pattern = self._normalize_address_regex(pattern)
            match = re.search(pattern, out)
            if match:
                addresses[label] = match.group(1)

        if "TARGET" not in addresses or "SAFETY_RULES" not in addresses:
            return self._attack_result(
                False,
                f"Critical addresses missing. Output:\n{out}",
                infra_broken=True,
                deployment_success=True,
                address_parse_success=False,
                foundry_summary={"deploy": deploy_summary},
            )

        print(f"      Mapped: {json.dumps(addresses, indent=2)}")

        # 4. Safety Check Script
        safety_script = self._generate_safety_check_script(addresses, safety_check_logic)
        self.save_file("script/CheckSafety.s.sol", safety_script)

        # 5. Configure the compact agent set.
        logs = []
        traces = []
        attack_episodes = 0
        breaches = 0
        agent_steps = 0
        malicious_steps = 0
        front_run_attempts = 0
        back_run_attempts = 0
        schedule_phases = []
        honest_intents = []
        history = []
        realtime_log_path = os.path.join(self.root_dir, "realtime_simulation.log")
        open(realtime_log_path, 'w').close()

        lifecycle_setup = self._run_lifecycle_setup(addresses, interface_desc)
        if lifecycle_setup.get("enabled"):
            history.append(
                "Lifecycle setup before adversarial scheduling: "
                f"{lifecycle_setup.get('status')} | {lifecycle_setup.get('action', '')[:500]}"
            )
            logs.append(
                "[LifecycleSetup]\n"
                f"Status: {lifecycle_setup.get('status')}\n"
                f"Action: {lifecycle_setup.get('action')}\n"
                f"LogTail:\n{lifecycle_setup.get('log_tail', '')}"
            )
            if self.event_logger:
                self.event_logger.emit("lifecycle_setup_result", lifecycle_setup)
            if lifecycle_setup.get("success"):
                self._ensure_core_sources_present(("src/Target.sol", "src/Mocks.sol", "src/SafetyRules.sol", "script/CheckSafety.s.sol"))
                chk_cmd = f"forge script script/CheckSafety.s.sol --rpc-url {self.RPC_URL}"
                s_out, s_err, s_code = self._run_command(
                    chk_cmd,
                    label="safety_check_lifecycle_setup",
                    kind="foundry_safety_check",
                    timeout_s=self._safety_check_timeout(),
                )
                setup_safety_summary = parse_foundry_output(s_out, s_err, s_code)
                lifecycle_setup["safety_check_pass"] = s_code == 0
                lifecycle_setup["safety_check_summary"] = setup_safety_summary
                if s_code != 0:
                    if self._command_timed_out(s_err):
                        timeout_trace = (
                            "Safety check timed out during lifecycle setup; treating as infrastructure timeout, "
                            "not as a semantic safety violation."
                        )
                        return self._attack_result(
                            False,
                            "\n".join(logs + [timeout_trace]),
                            infra_broken=True,
                            deployment_success=True,
                            address_parse_success=True,
                            lifecycle_setup=lifecycle_setup,
                            foundry_summary={"deploy": deploy_summary, "lifecycle_setup_safety": setup_safety_summary},
                        )
                    violation_reason = self._extract_revert_reason(s_err)
                    breach_trace = (
                        "!!! SAFETY VIOLATION DETECTED DURING LIFECYCLE SETUP !!!\n"
                        f"Reason: {violation_reason}\nRaw: {s_err}"
                    )
                    return self._attack_result(
                        False,
                        "\n".join(logs + [breach_trace]),
                        attack_episodes=0,
                        breaches=1,
                        traces=[breach_trace],
                        deployment_success=True,
                        address_parse_success=True,
                        lifecycle_setup=lifecycle_setup,
                        foundry_summary={"deploy": deploy_summary, "lifecycle_setup_safety": setup_safety_summary},
                    )

        honest_agents = []
        malicious_agents = []

        for name, data in self.accounts_config.items():
            if name == "Deployer": continue
            agent_obj = {
                "name": name,
                "role": data["role"],
                "key": data["private_key"],
                "address": data.get("address"),
            }
            if data["role"] == "HONEST_USER":
                honest_agents.append(agent_obj)
            elif data["role"] == "MALICIOUS_ATTACKER":
                malicious_agents.append(agent_obj)

        honest_limit = max(1, int(os.getenv("LEVER_HONEST_AGENTS", "2")))
        malicious_limit = max(0, int(os.getenv("LEVER_MALICIOUS_AGENTS", "2")))
        honest_agents = honest_agents[:honest_limit]
        malicious_agents = malicious_agents[:malicious_limit]
        print(f"   👥 Agents Optimized: {len(honest_agents)} Honest, {len(malicious_agents)} Malicious")

        # Give each agent an independent string-based memory.
        agent_memories = {a['name']: "" for a in honest_agents + malicious_agents}

        # POMG-style adversarial scheduling: attackers observe a pending honest
        # intent, can front-run, then the honest action executes, then attackers
        # can back-run. This better matches the paper's interleaved scheduler.
        TOTAL_ROUNDS = max(1, self.attack_rounds)

        for round_num in range(1, TOTAL_ROUNDS + 1):
            print(f"\n   ⚔️  [Round {round_num}/{TOTAL_ROUNDS}] Simulation Starting...")
            random.shuffle(honest_agents)

            for honest_idx, honest_agent in enumerate(honest_agents):
                current_memory = agent_memories.get(honest_agent['name'], "")
                print(f"      🧾 Planning pending honest intent: {honest_agent['name']} (Mem: {len(current_memory)} chars)")
                planned_action, planned_memory = self._get_llm_action_body(
                    honest_agent,
                    addresses,
                    interface_desc,
                    context_prompt=(
                        f"[HISTORY]\n{json.dumps(history[-5:]) if history else 'No previous actions.'}\n"
                        f"[CURRENT STATE]\nRound: {round_num}\n"
                        f"Your Identity: {honest_agent['name']} ({honest_agent['role']})\n"
                        f"{self._build_phase_context('honest_intent_planning')}"
                    ),
                    allow_skip=False,
                    previous_memory=current_memory,
                )
                if planned_memory:
                    agent_memories[honest_agent['name']] = planned_memory

                pending_intent = {
                    "round": round_num,
                    "agent": honest_agent["name"],
                    "intent": self._short_intent(planned_action),
                }
                honest_intents.append(pending_intent)
                history.append(f"Round {round_num}: {honest_agent['name']} announced pending intent.")

                phase_plan = [
                    ("front_run", malicious_agents),
                    ("honest_action", [honest_agent]),
                    ("back_run", malicious_agents),
                ]

                for phase_name, phase_agents in phase_plan:
                    for agent in phase_agents:
                        is_malicious = (agent['role'] == "MALICIOUS_ATTACKER")
                        allow_skip = is_malicious
                        phase_context = self._build_phase_context(
                            phase_name,
                            pending_intent if is_malicious else None,
                        )
                        schedule_phases.append({
                            "round": round_num,
                            "honest_intent_index": honest_idx,
                            "phase": phase_name,
                            "agent": agent["name"],
                            "role": agent["role"],
                        })

                        if phase_name == "front_run":
                            front_run_attempts += 1
                        elif phase_name == "back_run":
                            back_run_attempts += 1

                        current_memory = agent_memories.get(agent['name'], "")
                        print(f"      👉 {phase_name}: {agent['name']} ({agent['role']}) acting... (Mem: {len(current_memory)} chars)")

                        initial_action = planned_action if phase_name == "honest_action" else None
                        success, output_log, action_body, new_memory = self._run_agent_with_retry(
                            agent,
                            round_num,
                            addresses,
                            interface_desc,
                            history,
                            allow_skip=allow_skip,
                            previous_memory=current_memory,
                            phase_context=phase_context,
                            initial_action_body=initial_action,
                        )

                        if new_memory:
                            agent_memories[agent['name']] = new_memory
                            print(f"         🧠 Memory Updated.")

                        if action_body == "// SKIP":
                            print(f"         💤 {agent['name']} decided to WAIT.")
                            history.append(f"Round {round_num}-{phase_name}: {agent['name']} waited.")
                            continue

                        agent_steps += 1
                        if is_malicious:
                            malicious_steps += 1
                            attack_episodes += 1

                        script_section = f"📜 [FINAL SCRIPT]:\n{'-'*40}\n{action_body}\n{'-'*40}"
                        log_entry = (
                            f"[Round {round_num}][Phase {phase_name}][{agent['name']}] Result: {success}\n"
                            f"PendingIntent:\n{json.dumps(pending_intent, indent=2)}\n"
                            f"Script:\n{script_section}\nTrace:\n{output_log}\n{'-'*40}"
                        )
                        self._log_step(log_entry)

                        if success:
                            action_status, action_message = self._describe_action_outcome(output_log)
                            print(f"         {action_message}")
                            if self.event_logger:
                                self.event_logger.emit(
                                    "agent_action_outcome",
                                    {
                                        "round": round_num,
                                        "phase": phase_name,
                                        "agent": agent["name"],
                                        "role": agent["role"],
                                        "status": action_status,
                                    },
                                )
                            logs.append(
                                f"[Round {round_num}][Phase {phase_name}][{agent['name']}] {action_status.upper()}:\n"
                                f"[PENDING INTENT]: {json.dumps(pending_intent)}\n"
                                f"{script_section}\n[TRACE]:\n{output_log}"
                            )
                            if action_status == "confirmed":
                                history.append(f"Round {round_num}-{phase_name}: {agent['name']} executed a confirmed transaction.")
                            elif action_status == "partial_effect_observed":
                                history.append(f"Round {round_num}-{phase_name}: {agent['name']} produced unconfirmed semantic effect evidence; next action should isolate the same objective cleanly.")
                            elif action_status == "simulated_broadcast_failed":
                                history.append(f"Round {round_num}-{phase_name}: {agent['name']} simulated an action, but broadcast failed.")
                            else:
                                history.append(f"Round {round_num}-{phase_name}: {agent['name']} attempted an action that reverted or was not confirmed.")

                            chk_cmd = f"forge script script/CheckSafety.s.sol --rpc-url {self.RPC_URL}"
                            s_out, s_err, s_code = self._run_command(
                                chk_cmd,
                                label=f"safety_check_r{round_num}_{phase_name}_{agent['name']}",
                                kind="foundry_safety_check",
                                timeout_s=self._safety_check_timeout(),
                            )
                            safety_summary = parse_foundry_output(s_out, s_err, s_code)

                            if s_code != 0:
                                if self._command_timed_out(s_err):
                                    timeout_trace = (
                                        "Safety check timed out after an agent action; treating as infrastructure timeout, "
                                        "not as a semantic safety violation."
                                    )
                                    return self._attack_result(
                                        False,
                                        "\n".join(logs + [timeout_trace]),
                                        attack_episodes=attack_episodes,
                                        breaches=breaches,
                                        traces=traces,
                                        infra_broken=True,
                                        deployment_success=True,
                                        address_parse_success=True,
                                        agent_steps=agent_steps,
                                        malicious_steps=malicious_steps,
                                        total_rounds=TOTAL_ROUNDS,
                                        schedule_phases=schedule_phases,
                                        honest_intents=honest_intents,
                                        front_run_attempts=front_run_attempts,
                                        back_run_attempts=back_run_attempts,
                                        lifecycle_setup=lifecycle_setup,
                                        foundry_summary={"last_safety_check": safety_summary},
                                    )
                                print(f"      🚨 INVARIANT VIOLATED immediately after {agent['name']}'s action!")
                                violation_reason = self._extract_revert_reason(s_err)
                                breach_trace = (
                                    "!!! SAFETY VIOLATION DETECTED !!!\n"
                                    f"SchedulerPhase: {phase_name}\n"
                                    f"PendingIntent: {json.dumps(pending_intent)}\n"
                                    f"Reason: {violation_reason}\nRaw: {s_err}"
                                )
                                breaches += 1
                                logs.append(breach_trace)
                                traces.append(breach_trace)
                                return self._attack_result(
                                    False,
                                    "\n".join(logs),
                                    attack_episodes=attack_episodes,
                                    breaches=breaches,
                                    traces=traces,
                                    deployment_success=True,
                                    address_parse_success=True,
                                    agent_steps=agent_steps,
                                    malicious_steps=malicious_steps,
                                    total_rounds=TOTAL_ROUNDS,
                                    schedule_phases=schedule_phases,
                                    honest_intents=honest_intents,
                                    front_run_attempts=front_run_attempts,
                                    back_run_attempts=back_run_attempts,
                                    lifecycle_setup=lifecycle_setup,
                                    foundry_summary={"last_safety_check": safety_summary},
                                )
                            else:
                                print(f"      ✅ System remains safe.")
                        else:
                            print(f"         ❌ Action Failed!")
                            logs.append(f"[Round {round_num}][Phase {phase_name}][{agent['name']}] FAILED:\n{script_section}\nERROR:\n{output_log}")
                            history.append(f"Round {round_num}-{phase_name}: {agent['name']} failed.")

        print("\n   🏁 Simulation Ended. Final Safety Check passed.")
        return self._attack_result(
            True,
            "\n".join(logs),
            attack_episodes=attack_episodes,
            breaches=breaches,
            traces=traces,
            deployment_success=True,
            address_parse_success=True,
            agent_steps=agent_steps,
            malicious_steps=malicious_steps,
            total_rounds=TOTAL_ROUNDS,
            schedule_phases=schedule_phases,
            honest_intents=honest_intents,
            front_run_attempts=front_run_attempts,
            back_run_attempts=back_run_attempts,
            lifecycle_setup=lifecycle_setup,
            foundry_summary={"deploy": deploy_summary},
        )

    # Accept and return the agent's memory across rounds.
    def _run_agent_with_retry(
        self,
        agent,
        round_num,
        addresses,
        interface_desc,
        history,
        allow_skip=False,
        previous_memory="",
        phase_context="",
        initial_action_body=None,
    ):
        max_retries = 3
        current_retry = 0
        last_error = ""
        action_body = ""

        # Preserve the prior memory by default.
        final_memory = previous_memory

        context_prompt = f"""
        [HISTORY]
        {json.dumps(history[-5:]) if history else "No previous actions."}

        [CURRENT STATE]
        Round: {round_num}
        Your Identity: {agent['name']} ({agent['role']})

        {phase_context}
        """

        while current_retry < max_retries:
            try:
                # 1. Obtain the generated script and updated memory.
                if initial_action_body is not None and current_retry == 0:
                    action_body, next_memory = initial_action_body, previous_memory
                else:
                    action_body, next_memory = self._get_llm_action_body(
                        agent, addresses, interface_desc, context_prompt, last_error,
                        allow_skip, previous_memory
                    )

                if allow_skip and "// SKIP" in action_body:
                    return True, "SKIPPED", "// SKIP", next_memory

                print(f"         📜 [EXECUTING SCRIPT (Attempt {current_retry + 1})]:\n{'-'*40}\n{action_body}\n{'-'*40}")

                # 2. Wrap the script and write it to disk.
                self._ensure_core_sources_present()
                full_script = self._wrap_action_script(action_body, agent['key'])
                script_name = f"Action_{agent['name']}_{round_num}_{current_retry}_{int(time.time())}.s.sol"
                script_path = os.path.join(self.script_dir, script_name)

                with open(script_path, "w") as f:
                    f.write(full_script)

                # 3. Execute the Forge script.
                self._ensure_core_sources_present()
                self._fund_actor_account(agent)
                cmd = f"forge script script/{script_name} --tc ActionScript --broadcast --rpc-url {self.RPC_URL} -vvvv"
                out, err, code = self._run_command(
                    cmd,
                    label=f"agent_action_{agent['name']}_r{round_num}_try{current_retry + 1}",
                    kind="foundry_agent_action",
                )

                # =================================================================================
                # Distinguish compilation errors from transaction reverts.
                # =================================================================================

                # Case A: execution completed successfully.
                if code == 0:
                    trace_summary = self._extract_trace_section(out)
                    foundry_summary = parse_foundry_output(out, err, code)
                    status = foundry_summary.get("execution_status", "broadcast_confirmed")
                    self._archive_action_script(script_path)
                    return True, f"[ACTION_CONFIRMED_ONCHAIN status={status}]\n{trace_summary}", action_body, next_memory

                # Case B: check for a compilation or syntax error.
                # Foundry compiler errors usually contain "Compiler run failed" or "Error (".
                full_output = out + "\n" + err
                is_compilation_error = "Compiler run failed" in full_output or "Error (" in full_output or "Expected" in full_output

                if not is_compilation_error:
                    # Case C: the script compiled, but the transaction reverted.
                    # This often means the attack was blocked; record it without retrying.
                    foundry_summary = parse_foundry_output(out, err, code)
                    execution_status = foundry_summary.get("execution_status", "command_failed")
                    partial_effect = self._has_semantic_effect_evidence(out)
                    if partial_effect:
                        print("         🟡 Unconfirmed script contains state-changing semantic evidence.")
                    if execution_status in {"broadcast_failed_insufficient_funds", "simulation_success_broadcast_failed"}:
                        print(f"         ⚠️ Simulation succeeded but broadcast failed ({execution_status}). Stopping retries.")
                    else:
                        print(f"         🛡️  Transaction reverted or simulation failed ({execution_status}). Stopping retries.")
                    trace_summary = self._extract_trace_section(out)

                    # Return True so the experiment continues, but preserve the
                    # exact execution status for logs and later triage.
                    if partial_effect:
                        marker = "[ACTION_PARTIAL_EFFECT_OBSERVED]"
                    elif execution_status in {"broadcast_failed_insufficient_funds", "simulation_success_broadcast_failed"}:
                        marker = "[ACTION_SIMULATED_BROADCAST_FAILED]"
                    else:
                        marker = "[ACTION_SIMULATION_FAILED_OR_REVERTED]"
                    self._archive_action_script(script_path)
                    return True, f"{marker} status={execution_status}\n{trace_summary}", action_body, next_memory

                # Case D: generated code failed to compile; enter the retry loop.
                current_retry += 1
                print(f"         ⚠️ Syntax/Compiler Error (Attempt {current_retry}/{max_retries}). Retrying...")
                print(f"         🗑️  Deleting broken script: {script_name}")

                try: os.remove(script_path)
                except: pass

                # Send compiler feedback to the model for repair.
                full_error_trace = f"STDOUT:\n{out}\nSTDERR:\n{err}"
                last_error = f"Previous script failed to compile.\n{full_error_trace}\n\nFIX the syntax errors."

            except FatalLLMError:
                raise
            except Exception as e:
                print(f"         ❌ System Error: {e}")
                current_retry += 1

        # Return after exhausting retries, usually because syntax errors remain.
        return False, f"Failed after {max_retries} retries (Syntax Errors).\nLAST ERROR:\n{last_error}", action_body, final_memory

    def _extract_trace_section(self, output: str) -> str:
        if "Traces:" in output:
            return "Traces:" + output.split("Traces:")[1]
        return output[-2000:]

    def _extract_revert_reason(self, error_output: str) -> str:
        match = re.search(r"Error: script failed: (.*)", error_output)
        if match: return match.group(1)
        if "Revert reason:" in error_output: return error_output.split("Revert reason:")[1].strip()
        return "Unknown Revert"

    # Build the action prompt and parse the injected memory block.
    def _get_llm_action_body(self, agent, addresses, interface_desc, context_prompt, error_feedback="", allow_skip=False, previous_memory=""):

        # 1. Build the memory section.
        memory_section = ""
        if previous_memory:
            memory_section = f"""
            [YOUR PRIVATE MEMORY]
            In the previous round, you noted:
            "{previous_memory}"

            CONTINUE your strategy from this state. If you deployed a malicious contract before, use its address directly.
            """
        else:
            memory_section = "[YOUR PRIVATE MEMORY]\nNo memory. Start a new strategy."

        # 2. Build the optional skip instruction.
        skip_instruction = ""
        if allow_skip:
            skip_instruction = """
            - If you want to WAIT for other users to interact first, output ONLY: // SKIP
            """

        system_prompt = f"You are {agent['name']}, a {agent['role']} in a DeFi simulation."
        lifecycle_mode_instruction = ""
        if agent.get("role") == "LIFECYCLE_SETUP_OPERATOR":
            lifecycle_mode_instruction = """
        [LIFECYCLE SETUP MODE - OVERRIDES ATTACK GUIDANCE]
        You are not an attacker or an honest user. Your only job is to make later
        user/attacker actions reachable from a valid business phase.
        - Do not mint, buy, deposit, withdraw, claim, redeem, distribute funds, send
          funds, drain assets, or stress-repeat any business action.
        - Do not change caps, quotas, reserves, max supply, per-user limits, or other
          safety thresholds unless the interface makes that mandatory for initialization.
        - For mutually exclusive phases, choose one useful phase and stop. Examples:
          open presale OR public sale, not both; move to Active OR Distribution, not both.
        - Do not include mutating calls you expect to revert, even inside try/catch.
        - A good lifecycle script ends immediately after successful open/init/whitelist/
          price setup and simple read-only logs.
            """

        # 3. Add prompt safeguards against import-related naming conflicts.
        user_prompt = f"""
        Write a Foundry script to interact with the target.

        [ENVIRONMENT]
        Target Address: {addresses.get('TARGET')}
        All Addresses: {json.dumps(addresses)}

        [GLOBAL CONTEXT - READ CAREFULLY]
        The script execution environment AUTOMATICALLY imports:
        1. `src/Target.sol` (business logic and any target-local interfaces)
        2. `src/Mocks.sol` (MockERC20, MockERC721 when generated)
        3. A minimal compatibility `IERC20` / `IERC721` interface ONLY when your action code uses that name and the imported files do not already define it.

        {lifecycle_mode_instruction}

        [⛔️ COMPILATION SAFETY RULES]
        1. **NO INTERFACE REDEFINITION**:
           - **DO NOT** write `interface IERC20 {{ ... }}` or `contract Ownable {{ ... }}`.
           - The wrapper/imports provide common interfaces when available.
           - **Use imported or wrapper-provided names directly**: e.g., `IERC20(token).balanceOf(...)`.

        2. **CUSTOM INTERFACES**:
           - If you strictly need an interface NOT present in the Target, **YOU MUST PREFIX IT**.
           - Example: `interface I_Hack_Liquidity {{ ... }}` (Use "I_Hack_" prefix).
           - NEVER use generic names like `IMinter` or `IToken` if they might exist in Mocks.
           - If a helper type contains only external function signatures and no implementations,
             it MUST be declared as `interface`, not `contract`. Do not write
             `contract I_Hack_X {{ function f() external; }}`.

        3. **NO `address(this)` IN ACTION**:
           - Inside the `/* [ACTION] */` block (which goes into `run()`), **DO NOT use `address(this)`**.
           - Use `actorEOA` / `currentActor` / `attackerEOA` for your own externally-owned account, or deploy a Helper contract if you need an address with code.
           - Do NOT infer your identity from `msg.sender` inside the script body; the wrapper supplies your real actor address.
           - Do NOT use low-level `targetAddress.call{{value: amount}}(...)` for payable user actions; use typed calls such as `target.mint{{value: amount}}(...)` so the broadcast actor is the payer.
           - Prefer direct typed calls from `actorEOA`. Only use helper contracts when the target interaction genuinely requires contract code (for example reentrancy callbacks).
             Helper-funded payable calls are fragile in Foundry script broadcasts; if the same behavior can be tested with a direct EOA typed call, use the direct call.

        4. **OUTPUT SHAPE**:
           - Do NOT write `pragma`, `import`, `contract ActionScript`, or a complete wrapper contract.
           - In `/* [HELPER] */`, define only helper contracts/interfaces/libraries. Do NOT define free functions there.
           - In `/* [ACTION] */`, write only the statements that belong inside `run()`. Do NOT write `function run() external {{ ... }}`.
           - If a helper needs receive/fallback declarations, write `receive() external payable {{ ... }}` or `fallback() external payable {{ ... }}` in a contract. Do not put receive/fallback declarations inside an interface unless the target interface explicitly needs them.

        [CRITICAL: VARIABLE DECLARATION]
        The variables `actorEOA`, `currentActor`, `attackerEOA`, `actorPrivateKey`, and `agentPrivateKey` are provided by the wrapper.
        The variables `target` or token addresses are NOT global.
        **YOU MUST DECLARE THEM** inside the `/* [ACTION] */` block before using them.
        - The target contract is NOT named `Target` unless the imported source explicitly declares `contract Target`.
          Use the actual contract name from [INTERFACE DESCRIPTION] (for example `LiquidityPool target = LiquidityPool(targetAddress);`)
          or use a prefixed interface such as `I_Hack_Target target = I_Hack_Target(targetAddress);`.
          NEVER write `Target target = Target(...)` unless `contract Target` exists in `src/Target.sol`.

        [CRITICAL: EXECUTION ROBUSTNESS]
        1. **ALWAYS USE TRY/CATCH**:
           - When calling functions on the Target (especially attacks like `changeAdmin`, `transfer`, etc.), they might REVERT if the contract is secure.
           - **DO NOT** let the script crash.
           - Wrap ALL attack calls in `try/catch` blocks.

           **BAD (Crashing Script):**
           target.changeAdmin(attacker);

           **GOOD (Robust Script):**
           try target.changeAdmin(attacker) {{
               console.log("Success: Admin changed");
           }} catch Error(string memory reason) {{
               reason;
               console.log("Failed: Admin change");
           }} catch {{
               console.log("Failed: Unknown reason");
           }}
        2. **KEEP EACH ACTION SMALL AND SEMANTICALLY VALID**:
           - Prefer the real target ABI from [INTERFACE DESCRIPTION]. Do not spend the whole action guessing unrelated admin/proxy/ERC20 method names.
           - Pick ONE concrete objective for this action (for example: deposit LP into pid 0, claim reward, emergency withdraw, or one admin takeover probe).
           - Choose exactly ONE action mode:
             MODE A: positive semantic path. Perform setup reads, then one useful successful business path or one focused repeated-call sequence for a semantic obligation.
             MODE B: authorization/revert probe. Test one privileged or blocked call and do not perform unrelated useful state changes.
             MODE C: read-only observation. Only inspect state and plan the next focused action in memory.
           - Do not mix MODE A with MODE B in the same broadcast. If you expect a call to revert, it belongs in a separate MODE B action.
           - After the first successful state-changing semantic operation in MODE A, do not append unrelated privileged probes. Either stop after post-state reads, or repeat only the same operation when the semantic goal is replay/cap/consumption.
           - Bound stress quantities so evidence stays executable and compact. For mint/buy/deposit/claim loops, use small amounts (usually 1-5 units) and 2-3 repetitions. Do NOT mint/buy/deposit the entire remaining supply or a very large cap; if remaining supply is large, sample a small bounded amount and log the remaining capacity.
           - If a script logs a successful mint/deposit/withdraw/claim/distribution/proxy creation and then later performs an expected revert probe, the experiment will treat the script as lower-confidence evidence. Avoid that by ending cleanly after the semantic operation and post-state reads.
           - If a function requires a `pid`, call the pid-aware signature from the real ABI such as `deposit(uint256 pid, uint256 amount)`, not a guessed `deposit(uint256 amount)`.
           - If the actors need LP/reward/mock balances, mint or obtain them before deposit/stake when the provided mocks expose `mint(address,uint256)`.
           - Avoid large "kitchen sink" scripts. Large scripts often hit stack-too-deep and reduce useful state exploration.
        3. **SEMANTIC ATTACK OBJECTIVES**:
           - If [INTERFACE DESCRIPTION] includes `[SEMANTIC OBLIGATIONS]`, treat those as priority objectives for useful exploration.
           - Prefer valid positive-path setup before declaring a security path safe: open/initialize a phase when an authorized exposed function permits it, whitelist or fund actors when mocks/target functions expose that setup, then exercise the semantic action.
           - When the current state exposes only one reachable business phase, use the function for that phase instead of waiting for a different phase. Examples: if `presaleOpen == true` and the actor is whitelisted/allowlisted, use the presale/allowlist mint or buy function; if public sale is closed but distribution is active, use claim/distribute; if saleState/currentState is Active/Ongoing, use the ordinary user action for that state.
           - For honest MODE A actions, prefer proving the intended positive path is reachable before doing anything adversarial: read phase flags, read actor eligibility, then call the matching user function with exact payment/amount. If multiple phases exist, do not assume "public" is the only useful phase.
           - For liability/backing obligations, compare assets and claims before/after actions using public getters and `address(targetAddress).balance` when meaningful.
           - For per-user cap obligations, repeatedly call the same allowed mint/buy path from the same actor and log per-user balance or minted amount before/after.
           - For replay/consumption obligations, repeat the same distribution/claim/withdraw/settlement call from a valid state and log whether credits duplicate or source assets decrease.
           - For proxy/factory obligations, create the instance, verify it has code, then call at least one expected method on the created instance.
           - Keep conclusions calibrated: if setup is blocked by role/state and you cannot reach the positive path, record that in memory instead of treating the revert as proof of safety.
        4. **CONSOLE SAFETY**:
           - Use simple console calls only: `console.log("tag")`, `console.log("tag", value)`, `console.logAddress(addr)`, or `console.logUint(num)`.
           - Do NOT use multi-value `console.log("a", x, "b", y)`; this Foundry version may not support that overload.
           - Do NOT write `console.log(bytes(reason))` or print dynamic bytes/string values from catch blocks. Assign the reason to a dummy variable if needed and print a fixed tag.

        [CRITICAL: PAYABLE TARGET CASTS]
        - If Solidity says the target contract has a payable receive/fallback, cast addresses as payable:
          `TokenContract target = TokenContract(payable(targetAddress));`
        - Helper function parameters that are cast to such a target should be `address payable targetAddress`.
        - Do not write `target.fn{{value: 0}}(...)` for zero-payment calls. A zero-value call option still requires a payable function; use plain `target.fn(...)` unless ETH is actually being sent.
        - For payable business functions such as mint/buy/deposit/subscribe, do not call with zero value or arbitrary overpayment when a price/fee is knowable. First read public price-style getters/variables that are actually present in [INTERFACE DESCRIPTION] (`price`, `mintPrice`, `tokenPrice`, `fee`, `cost`, `getPrice`, etc.) or use explicit constructor/config values visible in the context; then call with exactly `unitPrice * quantity` (or the documented fixed fee). Do not invent a getter interface for a function that is not in the real ABI, because ABI-decoding an absent getter can crash the script.
        - A successful getter read of `0` is a valid exact price. Track whether the getter call succeeded with a boolean; do NOT treat `unitPrice == 0` as "price unavailable" and switch to a probe value. If the getter succeeds and returns 0, send exactly 0.
        - If the price cannot be read, use the constructor/setup/history value when available; otherwise try one small documented/common value once, log that it is a price probe, and keep the action bounded.

        [INTERFACE DESCRIPTION]
        {interface_desc}

        {context_prompt}

        {memory_section}

        {f"[ERROR CORRECTION] {error_feedback}" if error_feedback else ""}

        [FORMAT REQUIRED]
        /* [HELPER] */
        // Define Attacker contracts here.
        // Example: contract MalloryExploit {{ ... }}

        /* [ACTION] */
        // Write execution logic for `run()` here.
        // Example:
        // MalloryExploit exploit = new MalloryExploit(target);
        // exploit.attack();

        /* [MEMORY] */
        // Summary of what you did and plan for next round.

        {skip_instruction}
        """

        # 4. Query the model.
        if self.client is None:
            raise RuntimeError("No LLM API key configured. Set LLM_API_KEY or OPENAI_API_KEY.")
        response = self._chat_completion_with_retries(
            "agent_action",
            model=FLASH_MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            **self._llm_completion_kwargs(FLASH_MODEL_NAME),
        ).choices[0].message.content

        # 5. Normalize the model output.
        content = response

        # Remove <think> blocks before processing Markdown fences.
        if content:
            # re.DOTALL matches reasoning blocks that span multiple lines.
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)

        if "```" in content:
            # Use a general expression to cover fences such as ```json.
            content = re.sub(r"^```[a-zA-Z]*", "", content, flags=re.MULTILINE)
            content = re.sub(r"```$", "", content, flags=re.MULTILINE)

        content = content.strip()
        # 6. Parse the memory and code sections.
        # Default to old memory if parsing fails
        next_mem = previous_memory

        # Extract the memory section.
        if "/* [MEMORY] */" in content:
            parts = content.split("/* [MEMORY] */")
            content_without_mem = parts[0]
            # Treat the trailing section as memory.
            if len(parts) > 1:
                next_mem = parts[1].strip()
            content = content_without_mem # remaining is helper + action

        return content.strip(), next_mem

    def _wrap_action_script(self, body, private_key):
        # Default Logic to split HELPER and ACTION if present
        helper_code = ""
        action_code = body

        if "/* [HELPER] */" in body and "/* [ACTION] */" in body:
            parts = body.split("/* [ACTION] */")
            helper_code = parts[0].replace("/* [HELPER] */", "").strip()
            action_code = parts[1].strip()

        helper_code, action_code = self._sanitize_action_sections(helper_code, action_code)
        support_interfaces = self._action_support_interfaces(helper_code + "\n" + action_code)

        return f"""
        // SPDX-License-Identifier: MIT
        pragma solidity ^0.8.20;
        import "forge-std/Script.sol";
        import "../src/Target.sol";
        import "../src/Mocks.sol";

        // ================= COMPATIBILITY INTERFACES =========
        {support_interfaces}
        // ====================================================

        // ================= HELPER CONTRACTS =================
        {helper_code}
        // ====================================================

        contract ActionScript is Script {{
            function run() external {{
                uint256 key = {private_key};
                uint256 actorPrivateKey = key;
                uint256 agentPrivateKey = key;
                address payable actorEOA = payable(vm.addr(key));
                address payable currentActor = actorEOA;
                address payable attackerEOA = actorEOA;
                vm.deal(attackerEOA, 10000 ether);
                console.log("ACTION_ACTOR", actorEOA);
                vm.startBroadcast(key);

                // ================= ACTION LOGIC =================
                {action_code}
                // ================================================

                vm.stopBroadcast();
            }}
        }}
        """

    def _strip_solidity_preamble(self, code: str) -> str:
        lines = []
        for line in (code or "").splitlines():
            stripped = line.strip()
            if stripped.startswith("pragma solidity"):
                continue
            if stripped.startswith("import "):
                continue
            lines.append(line)
        return "\n".join(lines).strip()

    def _sanitize_deploy_script(self, code: str) -> str:
        code = code or ""
        def _empty_import_repl(match):
            body = match.group("body")
            body_without_comments = re.sub(r"/\*[\s\S]*?\*/|//[^\n]*", "", body).strip()
            if body_without_comments:
                return match.group(0)
            return f"import {match.group('path')};"

        code = re.sub(
            r'import\s*\{\s*(?P<body>[\s\S]*?)\s*\}\s*from\s*(?P<path>"[^"]+"|\'[^\']+\')\s*;',
            _empty_import_repl,
            code,
        )
        # `forge script --private-key` supplies the broadcaster. Generated deploy
        # scripts must not broadcast from arbitrary agent addresses such as Alice.
        return re.sub(r"\bvm\.startBroadcast\s*\([^;]*\)\s*;", "vm.startBroadcast();", code)

    def _normalize_address_regex(self, pattern: str) -> str:
        """Recover common LLM regex typos such as `0x[a-fA-F0-9]40`."""
        if not isinstance(pattern, str):
            return ""
        return re.sub(r"0x(?P<class>\[[^\]]+\])(?P<count>\d+)", r"0x\g<class>{\g<count>}", pattern)

    def _extract_function_body(self, code: str, function_name: str = "run") -> Optional[str]:
        match = re.search(rf"\bfunction\s+{re.escape(function_name)}\s*\([^)]*\)[^{{;]*{{", code or "")
        if not match:
            return None

        open_brace = code.find("{", match.start())
        depth = 0
        for idx in range(open_brace, len(code)):
            char = code[idx]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return code[open_brace + 1:idx].strip()
        return None

    def _remove_contracts_with_run(self, code: str) -> str:
        code = code or ""
        run_match = re.search(r"\bcontract\s+\w+[^{}]*{[\s\S]*?\bfunction\s+run\s*\(", code)
        if not run_match:
            return code

        contract_start = code.rfind("contract", 0, run_match.start() + len(run_match.group(0)))
        open_brace = code.find("{", contract_start)
        if contract_start < 0 or open_brace < 0:
            return code

        depth = 0
        for idx in range(open_brace, len(code)):
            char = code[idx]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return (code[:contract_start] + code[idx + 1:]).strip()
        return code

    def _remove_free_run_function(self, code: str) -> str:
        code = code or ""
        match = re.search(r"\bfunction\s+run\s*\([^)]*\)[^{;]*{", code)
        if not match:
            return code

        open_brace = code.find("{", match.start())
        depth = 0
        for idx in range(open_brace, len(code)):
            char = code[idx]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return (code[:match.start()] + code[idx + 1:]).strip()
        return code

    def _sanitize_action_sections(self, helper_code: str, action_code: str) -> Tuple[str, str]:
        helper_code = self._strip_solidity_preamble(helper_code)
        action_code = self._strip_solidity_preamble(action_code)

        extracted_run = self._extract_function_body(action_code, "run")
        if extracted_run:
            helper_candidate = self._remove_free_run_function(self._remove_contracts_with_run(action_code))
            helper_code = "\n\n".join(
                part for part in [helper_code, helper_candidate] if part.strip()
            ).strip()
            action_code = extracted_run

        helper_code = self._strip_solidity_preamble(helper_code)
        action_code = self._strip_solidity_preamble(action_code)
        action_code = re.sub(r"^\s*vm\.startBroadcast\([^;]*\);\s*$", "", action_code, flags=re.MULTILINE)
        action_code = re.sub(r"^\s*vm\.stopBroadcast\(\);\s*$", "", action_code, flags=re.MULTILINE)
        action_code = re.sub(r"\bconsole\.log\s*\(\s*bytes\s*\([^)]*\)\s*\)\s*;", 'console.log("catch reason omitted");', action_code)
        helper_code = re.sub(r"\bconsole\.log\s*\(\s*bytes\s*\([^)]*\)\s*\)\s*;", 'console.log("catch reason omitted");', helper_code)
        action_code = self._strip_zero_value_call_options(action_code)
        helper_code = self._strip_zero_value_call_options(helper_code)
        action_code = self._rename_reserved_solidity_placeholders(action_code)
        helper_code = self._rename_reserved_solidity_placeholders(helper_code)
        action_code = self._normalize_action_actor_identity(action_code)
        action_code = self._normalize_target_type_alias(action_code)
        helper_code = self._normalize_target_type_alias(helper_code)
        enum_prefix = self._contract_state_type_prefix()
        action_code = self._normalize_contract_state_refs(action_code, enum_prefix)
        helper_code = self._normalize_contract_state_refs(helper_code, enum_prefix)
        for contract_name in self._payable_target_contracts():
            pattern = rf"\b{re.escape(contract_name)}\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)"
            replacement = rf"{contract_name}(payable(\1))"
            action_code = re.sub(pattern, replacement, action_code)
            helper_code = re.sub(pattern, replacement, helper_code)
        return helper_code, action_code

    def _strip_zero_value_call_options(self, code: str) -> str:
        return re.sub(
            r"(\.[A-Za-z_][A-Za-z0-9_]*)\s*\{\s*value\s*:\s*(?:0|uint256\s*\(\s*0\s*\)|0\s+(?:wei|ether))\s*\}\s*\(",
            r"\1(",
            code or "",
        )

    def _rename_reserved_solidity_placeholders(self, code: str) -> str:
        """Rename generated `_` variables in contexts where Solidity reserves it."""
        source = code or ""
        counter = 0

        def repl_returns(match: re.Match) -> str:
            nonlocal counter
            counter += 1
            return f"returns ({match.group('type').strip()} __leverIgnoredReturn{counter})"

        source = re.sub(
            r"returns\s*\(\s*(?P<type>[^(),;{}]+?)\s+_\s*\)",
            repl_returns,
            source,
        )
        source = re.sub(
            r"catch\s+Error\s*\(\s*string\s+memory\s+_\s*\)",
            "catch Error(string memory __leverIgnoredError)",
            source,
        )
        source = re.sub(
            r"catch\s*\(\s*bytes\s+memory\s+_\s*\)",
            "catch (bytes memory __leverIgnoredErrorBytes)",
            source,
        )
        return source

    def _target_contract_names(self) -> List[str]:
        code = self._strip_solidity_noise_for_scan(self._read_sim_source("Target.sol"))
        names = []
        for match in re.finditer(r"\b(?:(abstract)\s+)?contract\s+([A-Za-z_][A-Za-z0-9_]*)\b", code):
            if match.group(1):
                continue
            name = match.group(2)
            if name not in {"Ownable", "ReentrancyGuard"}:
                names.append(name)
        return names

    def _primary_target_contract_name(self) -> Optional[str]:
        names = self._target_contract_names()
        if not names:
            return None
        return names[-1]

    def _normalize_target_type_alias(self, code: str) -> str:
        code = code or ""
        names = self._target_contract_names()
        if "Target" in names:
            return code
        primary = self._primary_target_contract_name()
        if not primary:
            return code
        code = re.sub(r"\bTarget\s+([A-Za-z_][A-Za-z0-9_]*)\s*=", f"{primary} \\1 =", code)
        code = re.sub(r"\bTarget\s*\(", f"{primary}(", code)
        return code

    def _normalize_contract_state_refs(self, code: str, enum_prefix: str) -> str:
        code = code or ""
        enum_decl_marker = "__LEVER_ENUM_CONTRACT_STATE_DECL__"
        # Do not rewrite enum declarations themselves. Rewriting
        # `enum ContractState` into `enum ITarget.ContractState` produces invalid
        # Solidity, while references to the enum type should still be normalized.
        code = re.sub(r"\benum\s+ContractState\b", f"enum {enum_decl_marker}", code)
        code = re.sub(
            r"\b(?<![\.\w])ContractState\b",
            enum_prefix,
            code,
        )
        code = re.sub(
            r"\b(?<![\.\w])ContractState\s*\(",
            f"{enum_prefix}(",
            code,
        )
        code = re.sub(
            r"(?<![A-Za-z0-9_])IINFTDistributionContract\.ContractState\s*\(",
            f"{enum_prefix}(",
            code,
        )
        code = re.sub(
            r"(?<![A-Za-z0-9_])INFTDistributionContract\.ContractState\s*\(",
            f"{enum_prefix}(",
            code,
        )
        code = re.sub(
            r"(?<![A-Za-z0-9_])NFTDistributionContract\.ContractState\s*\(",
            f"{enum_prefix}(",
            code,
        )
        code = code.replace(f"enum {enum_decl_marker}", "enum ContractState")
        return code

    def _normalize_action_actor_identity(self, action_code: str) -> str:
        action_code = action_code or ""
        # In a Foundry Script, top-level `msg.sender` is the script caller/default
        # context, not necessarily the EOA used by `vm.startBroadcast(key)`.
        # The wrapper provides actorEOA as the current agent's configured EOA.
        return re.sub(r"\bmsg\.sender\b", "actorEOA", action_code)

    def _read_sim_source(self, filename: str) -> str:
        try:
            return open(os.path.join(self.src_dir, filename), "r", encoding="utf-8").read()
        except Exception:
            return ""

    def _strip_solidity_comments_for_scan(self, code: str) -> str:
        """Remove comments before regex-based Solidity declaration scans."""
        return re.sub(r"/\*[\s\S]*?\*/|//[^\n]*", "", code or "")

    def _strip_solidity_noise_for_scan(self, code: str) -> str:
        """Remove comments and string literals before declaration scans.

        Regex-based scans are only used for coarse Solidity declarations. String
        contents such as "Insufficient contract balance" must not be interpreted
        as a `contract balance` declaration.
        """
        code = self._strip_solidity_comments_for_scan(code)
        return re.sub(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'', '""', code)

    def _payable_target_contracts(self) -> List[str]:
        code = self._strip_solidity_noise_for_scan(self._read_sim_source("Target.sol"))
        payable_contracts = []
        contract_pattern = re.compile(r"\b(contract|interface)\s+([A-Za-z_][A-Za-z0-9_]*)[^{}]*{")
        for match in contract_pattern.finditer(code):
            kind, name = match.group(1), match.group(2)
            if kind != "contract":
                continue
            open_brace = code.find("{", match.end() - 1)
            if open_brace < 0:
                continue
            depth = 0
            end = None
            for idx in range(open_brace, len(code)):
                if code[idx] == "{":
                    depth += 1
                elif code[idx] == "}":
                    depth -= 1
                    if depth == 0:
                        end = idx
                        break
            body = code[open_brace + 1:end] if end else ""
            if re.search(r"\breceive\s*\([^)]*\)\s*external\s+payable\b", body) or re.search(r"\bfallback\s*\([^)]*\)\s*external\s+payable\b", body):
                payable_contracts.append(name)
        return payable_contracts

    def _action_support_interfaces(self, code: str) -> str:
        sources = "\n".join([self._read_sim_source("Target.sol"), self._read_sim_source("Mocks.sol")])
        snippets = []

        def missing_type(type_name: str) -> bool:
            return (
                re.search(rf"\b{re.escape(type_name)}\b", code or "") is not None
                and re.search(rf"\b(interface|contract)\s+{re.escape(type_name)}\b", sources) is None
            )

        if missing_type("IERC20"):
            snippets.append("""
interface IERC20 {
    function balanceOf(address account) external view returns (uint256);
    function transfer(address to, uint256 amount) external returns (bool);
    function transferFrom(address from, address to, uint256 amount) external returns (bool);
    function approve(address spender, uint256 amount) external returns (bool);
    function allowance(address owner, address spender) external view returns (uint256);
}
""".strip())

        if missing_type("IERC721"):
            snippets.append("""
interface IERC721 {
    function balanceOf(address owner) external view returns (uint256);
    function ownerOf(uint256 tokenId) external view returns (address);
    function transferFrom(address from, address to, uint256 tokenId) external;
    function approve(address to, uint256 tokenId) external;
    function getApproved(uint256 tokenId) external view returns (address);
    function isApprovedForAll(address owner, address operator) external view returns (bool);
}
""".strip())

        return "\n\n".join(snippets)

    def _contract_state_type_prefix(self) -> str:
        target_path = os.path.join(self.src_dir, "Target.sol")
        try:
            code = open(target_path, "r", encoding="utf-8").read()
        except Exception:
            code = ""
        if re.search(r"\binterface\s+INFTDistributionContract\b", code):
            return "INFTDistributionContract.ContractState"
        return "NFTDistributionContract.ContractState"

    def _normalize_safety_check_logic(self, check_logic: str) -> str:
        """Clean generated check snippets before embedding them in CheckSafety."""
        code = check_logic or ""
        return re.sub(
            r'(?P<quote>["\'])(?P<char>[^"\'\\])(?P=quote)\s*\.\s*bytes1\b',
            r'bytes1("\g<char>")',
            code,
        )

    def _normalize_safety_rules_code(self, safety_code: str) -> str:
        """Keep dynamic SafetyRules enumerable in local experiments.

        View-only safety referees sometimes enumerate production-sized token
        spaces such as TOKEN_LIMIT(). That is useful as an obligation signal,
        but too expensive for per-action dynamic checks. Bound those scans to
        a small deterministic window while leaving tracked-user checks intact.
        """
        code = safety_code or ""
        scan_limit = max(1, int(os.getenv("LEVER_SAFETY_SCAN_LIMIT", "256")))
        changed = False

        assignment_re = re.compile(
            r"^(\s*)uint256\s+([A-Za-z_]\w*)\s*=\s*([^;\n]*(?:TOKEN_LIMIT|tokenLimit|MAX_SUPPLY|maxSupply)\s*\(\)\s*);(\s*(?://.*)?)$"
        )
        lines = code.splitlines()
        normalized_lines = []
        for idx, line in enumerate(lines):
            normalized_lines.append(line)
            match = assignment_re.match(line)
            if not match:
                continue
            indent, var_name = match.group(1), match.group(2)
            next_line = lines[idx + 1].strip() if idx + 1 < len(lines) else ""
            if next_line.startswith(f"if ({var_name} > "):
                continue
            normalized_lines.append(f"{indent}if ({var_name} > {scan_limit}) {{")
            normalized_lines.append(f"{indent}    {var_name} = {scan_limit};")
            normalized_lines.append(f"{indent}}}")
            changed = True

        code = "\n".join(normalized_lines)

        def bound_inline(match: re.Match) -> str:
            nonlocal changed
            target_expr = match.group(1)
            changed = True
            return f"<= ({target_expr}.TOKEN_LIMIT() > {scan_limit} ? {scan_limit} : {target_expr}.TOKEN_LIMIT())"

        code = re.sub(
            r"<=\s*([A-Za-z_][A-Za-z0-9_\.]*)\.TOKEN_LIMIT\(\)",
            bound_inline,
            code,
        )

        if changed and self.event_logger:
            self.event_logger.emit(
                "safety_rules_bounded_scan_normalized",
                {"scan_limit": scan_limit},
            )
        return code

    def _generate_safety_check_script(self, addresses, check_logic):
        check_logic = self._normalize_safety_check_logic(check_logic)
        return f"""
        // SPDX-License-Identifier: MIT
        pragma solidity ^0.8.20;
        import "forge-std/Script.sol";
        import "../src/SafetyRules.sol";

        contract CheckSafety is Script {{
            function run() external view {{
                SafetyRules rules = SafetyRules({addresses.get('SAFETY_RULES')});
                {check_logic}
            }}
        }}
        """

    def run_foundry_fuzzing(self, test_source: str):
        test_rel_path = "src/Invariant.t.sol"
        test_path = os.path.join(self.src_dir, "Invariant.t.sol")
        self._ensure_core_sources_present()
        with open(test_path, "w") as f:
            f.write(test_source)
        print("   🌪️  [Mode 2] Running Foundry Invariant Campaign (Handler-Based)...")
        fuzz_runs = int(os.getenv("LEVER_FUZZ_RUNS", "500"))
        cmd = f"forge test --offline --skip script --match-path {test_rel_path} -vvvv --fuzz-runs {fuzz_runs}"
        out, err, code = self._run_command(cmd, label="foundry_fuzz", kind="foundry_fuzz")
        combined = f"{out}\n{err}"
        foundry_summary = parse_foundry_output(out, err, code)
        if code != 0:
            if "Compiler run failed" in combined:
                print("   ⚠️ Fuzzing SKIPPED due to Compilation Error.")
                return {
                    "safe": False,
                    "log": f"SKIPPED_COMPILATION_ERROR\n{combined}",
                    "foundry_compile_pass": False,
                    "foundry_test_pass": False,
                    "infra_broken": True,
                    "broken_invariants": [],
                    "fuzz_runs": 0,
                    "foundry_summary": foundry_summary,
                }
            print("   🚨 INVARIANT BROKEN! Foundry found a sequence that breaks safety.")
            fuzz_report = self._parse_fuzz_failure(out)
            return {
                "safe": False,
                "log": fuzz_report,
                "foundry_compile_pass": True,
                "foundry_test_pass": False,
                "infra_broken": False,
                "broken_invariants": self._extract_broken_invariants(out),
                "fuzz_runs": self._extract_fuzz_runs(combined),
                "foundry_summary": foundry_summary,
            }
        return {
            "safe": True,
            "log": self._clean_fuzz_output(out),
            "foundry_compile_pass": True,
            "foundry_test_pass": True,
            "infra_broken": False,
            "broken_invariants": [],
            "fuzz_runs": self._extract_fuzz_runs(combined),
            "foundry_summary": foundry_summary,
        }

    def check_fuzz_compilation(self, test_source: str) -> Tuple[bool, str]:
        test_rel_path = "src/Invariant.t.sol"
        test_path = os.path.join(self.src_dir, "Invariant.t.sol")
        required_fragments = ["contract Handler", "contract InvariantTest", "invariant_"]
        missing = [fragment for fragment in required_fragments if fragment not in (test_source or "")]
        if missing:
            return False, "Generated Invariant.t.sol is structurally incomplete. Missing: " + ", ".join(missing)
        self._ensure_core_sources_present()
        with open(test_path, "w") as f:
            f.write(test_source)
        cmd = f"forge test --offline --skip script --match-path {test_rel_path} --list"
        out, err, code = self._run_command(cmd, label="foundry_fuzz_compile", kind="foundry_fuzz_compile")
        combined_output = f"STDOUT:\n{out}\nSTDERR:\n{err}"
        if code != 0:
            return False, combined_output
        return True, combined_output

    def _temporarily_disable_other_test_files(self, keep_basename: str) -> List[Tuple[str, str]]:
        moved: List[Tuple[str, str]] = []
        try:
            entries = os.listdir(self.src_dir)
        except OSError:
            return moved
        for name in entries:
            if name == keep_basename or not name.endswith(".t.sol"):
                continue
            src = os.path.join(self.src_dir, name)
            if not os.path.isfile(src):
                continue
            dst = src + ".disabled"
            counter = 0
            while os.path.exists(dst):
                counter += 1
                dst = f"{src}.disabled.{counter}"
            os.replace(src, dst)
            moved.append((dst, src))
        return moved

    def _restore_disabled_test_files(self, moved: List[Tuple[str, str]]):
        for dst, src in reversed(moved):
            if os.path.exists(dst):
                os.replace(dst, src)

    def check_semantic_probe_compilation(self, test_source: str) -> Tuple[bool, str]:
        test_rel_path = "src/SemanticProbe.t.sol"
        test_path = os.path.join(self.src_dir, "SemanticProbe.t.sol")
        required_fragments = ["forge-std/Test.sol", "contract", "function test"]
        missing = [fragment for fragment in required_fragments if fragment not in (test_source or "")]
        if missing:
            return False, "Generated SemanticProbe.t.sol is structurally incomplete. Missing: " + ", ".join(missing)
        self._ensure_core_sources_present()
        with open(test_path, "w") as f:
            f.write(test_source)
        moved = self._temporarily_disable_other_test_files("SemanticProbe.t.sol")
        try:
            cmd = f"forge test --offline --skip script --match-path {test_rel_path} --list"
            out, err, code = self._run_command(cmd, label="semantic_probe_compile", kind="semantic_probe_compile")
            combined_output = f"STDOUT:\n{out}\nSTDERR:\n{err}"
            if code != 0:
                return False, combined_output
            return True, combined_output
        finally:
            self._restore_disabled_test_files(moved)

    def run_semantic_probe(self, test_source: str) -> Dict:
        test_rel_path = "src/SemanticProbe.t.sol"
        test_path = os.path.join(self.src_dir, "SemanticProbe.t.sol")
        self._ensure_core_sources_present()
        with open(test_path, "w") as f:
            f.write(test_source)
        print("   🧭 [Diagnostic] Running Semantic Probe Tests...")
        moved = self._temporarily_disable_other_test_files("SemanticProbe.t.sol")
        try:
            cmd = f"forge test --offline --skip script --match-path {test_rel_path} -vvvv"
            out, err, code = self._run_command(cmd, label="semantic_probe", kind="semantic_probe")
        finally:
            self._restore_disabled_test_files(moved)
        combined = f"{out}\n{err}"
        foundry_summary = parse_foundry_output(out, err, code)
        failures = sorted(set(foundry_summary.get("failed_tests") or []))
        if code != 0:
            infra_broken = "Compiler run failed" in combined or foundry_summary.get("execution_status") == "compiler_failed"
            if infra_broken:
                print("   ⚠️ Semantic probe skipped due to compilation error.")
                return {
                    "available": True,
                    "safe": None,
                    "semantic_probe_compile_pass": False,
                    "semantic_probe_test_pass": False,
                    "infra_broken": True,
                    "semantic_failures": [],
                    "log": f"SKIPPED_COMPILATION_ERROR\n{combined}",
                    "foundry_summary": foundry_summary,
                }

            print("   🟡 Semantic probe found requirement-level evidence.")
            return {
                "available": True,
                "safe": False,
                "semantic_probe_compile_pass": True,
                "semantic_probe_test_pass": False,
                "infra_broken": False,
                "semantic_failures": failures,
                "log": self._parse_fuzz_failure(out) if out else text_tail(combined, 6000),
                "foundry_summary": foundry_summary,
            }

        return {
            "available": True,
            "safe": True,
            "semantic_probe_compile_pass": True,
            "semantic_probe_test_pass": True,
            "infra_broken": False,
            "semantic_failures": [],
            "log": self._clean_fuzz_output(out),
            "foundry_summary": foundry_summary,
        }

    def _extract_broken_invariants(self, output: str) -> List[str]:
        if "Failing tests:" not in output and "[FAIL" not in output:
            return []
        failing_section = output.split("Failing tests:", 1)[1] if "Failing tests:" in output else output
        return sorted(set(re.findall(r"invariant_(\w+)\(\)\s+\(runs:", failing_section)))

    def _extract_fuzz_runs(self, output: str) -> int:
        runs = [int(x) for x in re.findall(r"runs:\s*(\d+)", output)]
        return sum(runs)

    def _parse_fuzz_failure(self, output: str) -> str:
        report = []
        invariant_match = re.search(r"invariant_(\w+)\(\)", output)
        if invariant_match:
            report.append(f"[BROKEN PROPERTY]: invariant_{invariant_match.group(1)}")
        else:
            report.append("[BROKEN PROPERTY]: Unknown")
        reason_match = re.search(r"\[FAIL\. Reason: (.*?)\]", output)
        if reason_match:
            report.append(f"[VIOLATION REASON]: {reason_match.group(1)}")
        if "Traces:" in output:
            trace_part = output.split("Traces:")[1]
            report.append(f"\n[ATTACK TRACE / COUNTEREXAMPLE]:\n{trace_part[:4000]}")
        else:
            report.append(f"\n[RAW OUTPUT TAIL]:\n{output[-2000:]}")
        return "\n".join(report)

    def initialize_environment(self, target_code: str, mocks_code: str):
        self._setup_fs()
        self.save_file("src/Target.sol", target_code)
        self.save_file("src/Mocks.sol", mocks_code)

    def save_file(self, relative_path: str, content: str):
        rel = relative_path.replace("\\", "/").lstrip("/")
        if not rel or ".." in rel.split("/"):
            raise ValueError(f"Unsafe simulation relative path: {relative_path!r}")
        if rel == "src/SafetyRules.sol":
            content = self._normalize_safety_rules_code(content)
        full_path = os.path.join(self.root_dir, rel)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w") as f:
            f.write(content)
        self._remember_core_source(relative_path, content)

    def check_compilation(self) -> Tuple[bool, str]:
        cmd = "forge build --offline --skip test"
        out, err, code = self._run_command(cmd, label="forge_build", kind="foundry_build")
        combined_output = f"STDOUT:\n{out}\nSTDERR:\n{err}"
        if code != 0:
            return False, combined_output
        return True, ""

    def check_script_validity(self, script_path: str) -> Tuple[bool, str]:
        return self.check_compilation()

    # Helper for compacting verbose Foundry output.
    def _clean_fuzz_output(self, raw_output: str) -> str:
        """
        Retain compilation messages, warnings, test results, and statistics from
        Foundry output while truncating verbose "Traces:" stacks.
        """
        cleaned_lines = []
        capture = True

        # Track whether the summary table has ended.
        seen_table_end = False

        for line in raw_output.split('\n'):
            # Stop collecting verbose stack details after the Traces section starts.
            if "Traces:" in line:
                cleaned_lines.append("\n[... Traces truncated for readability ...]")
                capture = False
                continue

            # A failure summary can occasionally appear after traces, but Foundry
            # normally emits Header -> Tests -> Table -> Traces, so truncation is sufficient.

            if capture:
                cleaned_lines.append(line)

        return "\n".join(cleaned_lines).strip()
