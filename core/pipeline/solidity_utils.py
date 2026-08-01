import re

def is_plausible_solidity_source(code: str) -> bool:
    if not code or not code.strip():
        return False
    return (
        "pragma solidity" in code
        and re.search(r"\b(contract|interface|library)\s+[A-Za-z_][A-Za-z0-9_]*", code) is not None
    )
