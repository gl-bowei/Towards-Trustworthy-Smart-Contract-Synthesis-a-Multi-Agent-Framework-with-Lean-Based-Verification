import re
from core.logger import GLOBAL_LOGGER

def normalize_generated_lean(lean_code: str) -> str:
    """Normalize generated Lean before validation/compilation.

    Keep this in sync with the lean-only runner so integrated and standalone
    formal checks do not diverge on mechanical syntax variants.
    """
    code = (lean_code or "").strip()
    if not code:
        return ""
    if "set_option autoImplicit false" not in code:
        code = re.sub(
            r"(?m)^(\s*import\s+Lean\s*)$",
            r"\1\nset_option autoImplicit false",
            code,
            count=1,
        )

    code = re.sub(r":=\s*:=\s*(?:by\s+)?sorry\b", ":= sorry", code)
    code = re.sub(r":=\s*by\s+sorry\b", ":= sorry", code)

    binder_renames = {
        "from": "fromAddr",
        "to": "toAddr",
        "type": "assetKind",
        "at": "assetKindKey",
    }
    names_to_rename = set()
    for match in re.finditer(r"\(([^()\n:]+):", code):
        for raw_name in match.group(1).strip().split():
            name = raw_name.strip("{}[]")
            if name in binder_renames:
                names_to_rename.add(name)
    for match in re.finditer(r"\bfun\s+([A-Za-z_][A-Za-z0-9_']*)\s*=>", code):
        name = match.group(1)
        if name in binder_renames:
            names_to_rename.add(name)
    for old in sorted(names_to_rename, key=len, reverse=True):
        new = binder_renames[old]
        code = re.sub(
            rf"(?<![A-Za-z0-9_']){re.escape(old)}(?![A-Za-z0-9_'])",
            new,
            code,
        )

    return code

def extract_theorems_info(lean_code: str):
    """
    Parses 'Definitions.lean' to find theorems.
    Returns a list of dicts: {'name': '...', 'full_text': '...'}
    """
    pattern = r"^\s*theorem\s+(\w+)[\s\S]*?:\s*=\s*sorry"
    matches = []
    for m in re.finditer(pattern, lean_code, re.MULTILINE):
        matches.append({
            'name': m.group(1),
            'full_text': m.group(0)
        })
    GLOBAL_LOGGER.log_system("REGEX_EXTRACTION", f"Found {len(matches)} theorems: {[m['name'] for m in matches]}")
    return matches
