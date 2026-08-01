import re
from typing import Dict

def extract_semantic_evidence_snippets(log_text: str, *, max_snippets: int = 18, window: int = 1200) -> str:
    """Keep high-signal semantic traces even when the full agent log is long."""
    if not log_text:
        return ""

    patterns = [
        r"mint\w*[_ ]?(?:success|ok)",
        r"(?:success|ok).*mint",
        r"(?:second|repeat|batch|extra).*mint",
        r"(?:presale|publicSale|saleState|whitelist|allowlist).*?(?:success|open|true|before|after)",
        r"(?:cap|limit|quota|per.?user|per.?wallet).*?(?:success|fail|revert|exceed|bypass)",
        r"balance(?:Of)?[_ ]?(?:before|after)",
        r"totalSupply[_ ]?(?:before|after|final)",
        r"(?:target|contract)_eth_(?:before|after|final)",
        r"(?:claim|withdraw|redeem|deposit|distribute|settle).*?(?:success|ok)",
        r"(?:success|ok).*?(?:claim|withdraw|redeem|deposit|distribute|settle)",
        r"proxy.*?(?:code|deploy|created|success|failed)",
        r"ACTION_(?:CONFIRMED_ONCHAIN|PARTIAL_EFFECT_OBSERVED|SIMULATED_BROADCAST_FAILED|SIMULATION_FAILED_OR_REVERTED)",
        r"N/A: SEMANTIC",
    ]

    spans = []
    for pattern in patterns:
        for match in re.finditer(pattern, log_text, flags=re.IGNORECASE):
            start = max(0, match.start() - window // 2)
            end = min(len(log_text), match.end() + window // 2)
            spans.append((start, end))
            if len(spans) >= max_snippets * 3:
                break
        if len(spans) >= max_snippets * 3:
            break

    merged = []
    for start, end in sorted(spans):
        if not merged or start > merged[-1][1] + 200:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)

    snippets = []
    for idx, (start, end) in enumerate(merged[:max_snippets], 1):
        snippet = log_text[start:end].strip()
        if snippet:
            snippets.append(f"--- semantic snippet {idx} [{start}:{end}] ---\n{snippet}")
    return "\n\n".join(snippets)

def attack_property_to_text(record: Dict) -> str:
    prop = record.get("property", record) if isinstance(record, dict) else {}
    if not isinstance(prop, dict):
        return str(record)
    property_text = prop.get("property_text", "").strip()
    invariant_template = prop.get("invariant_template", "").strip()
    source_trace = prop.get("source_trace", "").strip()
    parts = []
    if property_text:
        parts.append(property_text)
    if invariant_template:
        parts.append(f"Invariant Template: {invariant_template}")
    if source_trace:
        parts.append(f"Derived from attack trace: {source_trace[:500]}")
    if not parts:
        return ""
    return "[ATTACK-DERIVED PROPERTY] " + " | ".join(parts)
