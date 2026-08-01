import re
from typing import List


def validate_generated_lean(lean_code: str, theorem_budget: int = 0) -> List[str]:
    """Reject invalid or vacuous formalizer output before invoking the prover."""
    code = lean_code or ""
    errors: List[str] = []
    if "[SECTION: DEFINITIONS]" not in code or "[SECTION: THEOREMS]" not in code:
        errors.append("Missing required DEFINITIONS/THEOREMS section markers.")
    if "set_option autoImplicit false" not in code:
        errors.append("Missing `set_option autoImplicit false`.")
    if re.search(r"(?<![\w'])\?[A-Za-z_][A-Za-z0-9_']*", code):
        errors.append("Unresolved Lean metavariable hole found.")
    if re.search(r"\b(?:axiom|admit)\b", code):
        errors.append("Axiom/admit declarations are forbidden.")

    theorem_text = code.split("[SECTION: THEOREMS]", 1)[-1]
    theorems = re.findall(
        r"(?m)^\s*theorem\s+([A-Za-z_][A-Za-z0-9_']*)[\s\S]*?:\s*=\s*sorry",
        theorem_text,
    )
    if not theorems:
        errors.append("No theorem obligations ending in `:= sorry` were generated.")
    budget = max(0, int(theorem_budget or 0))
    if budget and len(theorems) > budget:
        errors.append(f"Theorem budget exceeded: {len(theorems)} > {budget}.")
    if re.search(r":\s*(?:True|False|False\s*=\s*False)\s*:=\s*sorry", theorem_text):
        errors.append("Vacuous theorem conclusion found.")
    return errors
