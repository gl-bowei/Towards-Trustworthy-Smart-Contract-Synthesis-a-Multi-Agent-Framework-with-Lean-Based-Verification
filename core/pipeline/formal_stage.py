from typing import List
from core.pipeline.collection_utils import append_unique

def record_formal_feedback(ctx, failures: List[str]):
    if not failures:
        return
    compact = "\n".join(f"- {failure}" for failure in failures[:5])
    if len(failures) > 5:
        compact += f"\n- ... and {len(failures) - 5} more formal failure(s)"

    constraint = (
        "Resolve formal verification failures. The next contract revision must preserve the "
        "selected safety properties and simplify/strengthen the implementation so these Lean "
        f"obligations can be proven:\n{compact}"
    )
    append_unique(ctx.feedback_constraints, constraint)
    ctx.checkpoint.save_stage("formal_failure_feedback", failures)

def prove_single_theorem_with_retry(prover_agent, verifier_tool, definitions_code, theorem_info, max_retries=5):
    """
    Run the generate, verify, and repair loop for one theorem.
    """
    thm_name = theorem_info['name']
    full_text_pattern = theorem_info['full_text']
    current_error = ""

    for attempt in range(max_retries):
        try:
            # 1. Ask the agent to generate a proof.
            proof_script = prover_agent.prove_theorem(
                definitions_code,
                full_text_pattern,
                error_feedback=current_error
            )

            # 2. Ensure that the generated proof is not empty.
            if not proof_script or len(proof_script.strip()) < 5:
                raise ValueError("Agent returned empty proof.")

            clean_proof = proof_script.strip()
            clean_proof_lower = clean_proof.lower()
            if "sorry" in clean_proof_lower or "admit" in clean_proof_lower:
                raise ValueError("Agent returned an incomplete proof containing 'sorry' or 'admit'.")

            if not clean_proof.startswith("by"):
                raise ValueError("Agent returned non-Lean proof text; expected a proof starting with `by`.")

            # 3. Construct the complete source file.
            if full_text_pattern not in definitions_code:
                 raise ValueError(f"Pattern '{full_text_pattern[:20]}...' not found in definitions.")

            target_proof_def = full_text_pattern.replace(":= sorry", f":=\n{proof_script}")
            full_file_content = definitions_code.replace(full_text_pattern, target_proof_def)

            fname = f"Proof_{thm_name}.lean"
            verifier_tool.save_file(fname, full_file_content)

            # 4. Verify the generated proof.
            errors = verifier_tool.verify_file(fname)

            if not errors:
                print(f"   ✅ {thm_name} PROVEN! (Attempt {attempt+1})")
                return (True, thm_name, None)

            # 5. Prepare error feedback for the next attempt.
            # Remove irrelevant warnings and retain actual errors.
            real_errors = [e for e in errors if "warning" not in e.lower()]
            if not real_errors: real_errors = errors

            short_error = "\n".join(real_errors[:5])
            current_error = f"Attempt {attempt+1} Code:\n{proof_script}\n\nCompiler Output:\n{short_error}"

            print(f"   ⚠️ {thm_name} Failed Attempt {attempt+1}. Retrying...")

        except Exception as e:
            current_error = f"System Exception: {str(e)}"
            print(f"   ⚠️ {thm_name} System Error: {e}")

    print(f"   ❌ {thm_name} Failed after {max_retries} attempts.")
    return (False, thm_name, current_error)
