import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from core.logger import GLOBAL_LOGGER
from core.structured_logging import JsonlEventLogger
from core.agents import format_semantic_obligations
from core.pipeline.collection_utils import append_unique
from core.pipeline.diagnostic_utils import attack_property_to_text, extract_semantic_evidence_snippets
from core.pipeline.dynamic_utils import restore_simulation_sources_for_dynamic_modes
from core.pipeline.formal_stage import prove_single_theorem_with_retry, record_formal_feedback
from core.pipeline.lean_utils import extract_theorems_info, normalize_generated_lean
from core.pipeline.lean_validation import validate_generated_lean
from core.pipeline.metrics import MetricsTracker
from core.pipeline.options import env_flag
from core.pipeline.slither import run_slither_check

def run_common_pipeline(
    ctx,
    agents,
    tools,
    audit_mode=False,
    skip_proofs=False,
    ablation_mode="full",
    user_targets=None,
    run_llm_audit=True,
    run_slither=True,
    run_non_lean=True,
    require_formal_success=False,
):
    """
    Run the common verification flow: Properties -> Formal -> Sim -> Analysis.
    Return True when the configured criteria pass, otherwise False.
    """
    tracker = MetricsTracker()
    verifier_enabled = (not skip_proofs) and ablation_mode != "no_verifier"
    attacker_enabled = ablation_mode != "no_attacker"
    # Preserve the original core policy: formal results are non-blocking by
    # default in mixed modes. lean_only opts in because it has no dynamic gate.
    formal_blocks_dynamic = bool(require_formal_success) or env_flag(
        "LEVER_FORMAL_BLOCKS_DYNAMIC", "0"
    )
    formal_feedback_iterates = bool(require_formal_success) or env_flag(
        "LEVER_FORMAL_FEEDBACK_ITERATES", "0"
    )
    tracker.set_mode_flags(ablation_mode, verifier_enabled, attacker_enabled)
    tracker.stats["formal_blocks_dynamic"] = formal_blocks_dynamic
    tracker.stats["formal_feedback_iterates"] = formal_feedback_iterates
    tracker.stats["llm_audit_enabled"] = run_llm_audit
    tracker.stats["slither_enabled"] = run_slither
    user_targets = user_targets or []
    event_logger = JsonlEventLogger(
        os.path.join(ctx.workspace, f"events_iter_{ctx.iteration}.jsonl"),
        metadata={
            "workspace": os.path.abspath(ctx.workspace),
            "iteration": ctx.iteration,
            "ablation_mode": ablation_mode,
        },
    )
    ctx.event_logger = event_logger
    if hasattr(tools.get("simulator"), "attach_event_logger"):
        tools["simulator"].attach_event_logger(event_logger)
    event_logger.emit("iteration_start", {
        "skip_proofs": skip_proofs,
        "verifier_enabled": verifier_enabled,
        "attacker_enabled": attacker_enabled,
        "formal_blocks_dynamic": formal_blocks_dynamic,
        "formal_feedback_iterates": formal_feedback_iterates,
        "run_llm_audit": run_llm_audit,
        "run_slither": run_slither,
    })
    # Step 2: Select Properties
    # Regenerate properties from the current code while preserving security patches.
    print("Selecting Verification Properties...")
    semantic_extractor = agents.get("semantic_extractor")
    if semantic_extractor:
        ctx.semantic_obligations = semantic_extractor.extract(ctx.solidity_code, user_targets)
        ctx.semantic_property_texts = format_semantic_obligations(ctx.semantic_obligations, include_low_confidence=False)
        ctx.semantic_guidance_texts = format_semantic_obligations(ctx.semantic_obligations, include_low_confidence=True)
        ctx.checkpoint.save_stage("semantic_obligations", ctx.semantic_obligations)
        ctx.checkpoint.save_stage("semantic_property_texts", ctx.semantic_property_texts)
        ctx.checkpoint.save_stage("semantic_guidance_texts", ctx.semantic_guidance_texts)
        tracker.stats["semantic_obligation_count"] = len(ctx.semantic_obligations)
        tracker.stats["semantic_obligations"] = ctx.semantic_obligations
        tracker.stats["semantic_property_texts"] = ctx.semantic_property_texts
        tracker.stats["semantic_guidance_texts"] = ctx.semantic_guidance_texts
        tracker.stats["semantic_confidence_note"] = (
            "Requirement-derived semantic obligations are inferred checks. "
            "Report them separately from hard paper metrics."
        )
        event_logger.emit("semantic_obligations_extracted", {
            "count": len(ctx.semantic_obligations),
            "obligations": ctx.semantic_obligations,
        })
        if ctx.semantic_obligations:
            names = ", ".join(o.get("id", "") for o in ctx.semantic_obligations)
            print(f"   Semantic obligations: {names}")
    preserved_attack_properties = []
    for record in ctx.attack_properties:
        text = attack_property_to_text(record)
        if text and text not in ctx.security_patches:
            preserved_attack_properties.append(text)
    ctx.properties = agents['selector'].select_properties(
        ctx.solidity_code,
        user_targets,
        ctx.semantic_property_texts + ctx.security_patches + preserved_attack_properties # Include accumulated security patches.
    )
    print(f"   Selected {len(ctx.properties)} Properties")
    ctx.checkpoint.save_stage("selected_properties", ctx.properties)
    event_logger.emit("properties_selected", {
        "count": len(ctx.properties),
        "properties": ctx.properties,
        "semantic_obligations": ctx.semantic_obligations,
        "preserved_attack_properties": preserved_attack_properties,
    })

    # Step 3: Formalize (Lean 4) & Statistics
    lean_theorems_count = 0
    lean_proven_count = 0
    definitions_code = ""
    lean_definitions_compiled = False
    formal_success = not verifier_enabled
    formal_failures = []
    if verifier_enabled and not ctx.properties:
        formal_failures.append("PropertySelector selected no verification properties.")

    if not verifier_enabled:
        print("   SKIPPING Formal Verification Phase (Verifier disabled by ablation/legacy flag)...")
    else:
        # Step 3: Formalize
        lean_attempts = 0
        compilation_error = ""

        print("Architecting Lean 4 Model...")
        while lean_attempts < 4:
            theorem_budget = max(0, int(os.getenv("LEVER_LEAN_THEOREM_BUDGET", "0") or 0))
            definitions_code = agents['formalizer'].formalize_definitions(
                ctx.solidity_code,
                ctx.properties,
                error_feedback=compilation_error,
                theorem_budget=theorem_budget,
            )
            definitions_code = normalize_generated_lean(definitions_code)
            validation_errors = []
            validation_errors = validate_generated_lean(definitions_code, theorem_budget)
            tools['verifier'].save_file("Definitions.lean", definitions_code)
            if validation_errors:
                errors = validation_errors
                GLOBAL_LOGGER.log_system("LEAN_VALIDATION_FAILED", "\n".join(validation_errors[:5]))
            else:
                errors = tools['verifier'].verify_file("Definitions.lean")

            if not errors:
                print("   Definitions compiled.")
                GLOBAL_LOGGER.log_system("COMPILATION_SUCCESS", "Definitions.lean compiled.")
                ctx.checkpoint.save_stage("definitions_code", definitions_code)
                lean_definitions_compiled = True
                break
            else:
                lean_attempts += 1
                compilation_error = "\n".join(errors[:5])
                if validation_errors:
                    print(f"   Lean Validation Error (Retrying {lean_attempts}/4)...")
                else:
                    print(f"   Lean Compile Error (Retrying {lean_attempts}/4)...")

        if not lean_definitions_compiled:
            formal_failures.append(f"Definitions.lean failed to compile: {compilation_error or 'unknown Lean error'}")

        # Step 4: Proving
        if lean_definitions_compiled and definitions_code and lean_attempts < 4:
            theorems_info = extract_theorems_info(definitions_code)
            lean_theorems_count = len(theorems_info)
            print(f"Identified {len(theorems_info)} theorems. Starting Parallel Proof Generation...")

            proof_results = []
            if not theorems_info and ctx.properties:
                formal_failures.append("Lean formalizer produced no theorem obligations for the selected properties.")
            else:
                with ThreadPoolExecutor(max_workers=5) as executor:
                    future_to_info = {
                        executor.submit(
                            prove_single_theorem_with_retry,
                            agents['prover'],
                            tools['verifier'],
                            definitions_code,
                            info
                        ): info
                        for info in theorems_info
                    }
                    for future in as_completed(future_to_info):
                        info = future_to_info[future]
                        try:
                            ok, thm_name, error = future.result(timeout=300)
                        except Exception as e:
                            ok, thm_name, error = False, info.get("name", "unknown"), str(e)
                            print(f"   Proof Error: {e}")

                        proof_results.append({
                            "theorem": thm_name,
                            "proven": ok,
                            "error": error,
                        })
                        if ok:
                            lean_proven_count += 1
                        else:
                            formal_failures.append(f"{thm_name}: {error or 'proof failed'}")

            ctx.checkpoint.save_stage("proof_results", proof_results)

        formal_success = (
            lean_definitions_compiled
            and lean_theorems_count > 0
            and lean_proven_count == lean_theorems_count
        )
        if not formal_success and (formal_blocks_dynamic or formal_feedback_iterates):
            record_formal_feedback(ctx, formal_failures)
        elif not formal_success:
            ctx.checkpoint.save_stage("formal_failures_nonblocking", formal_failures)

        # [Update Tracker with Lean Stats]
        tracker.update_lean_stats(
            lean_theorems_count,
            lean_proven_count,
            definitions_compiled=lean_definitions_compiled,
            failed_theorems=formal_failures,
            formal_success=formal_success,
        )
        event_logger.emit("lean_result", {
            "definitions_compiled": lean_definitions_compiled,
            "theorems_total": lean_theorems_count,
            "theorems_proven": lean_proven_count,
            "failed_theorems": formal_failures,
            "formal_success": formal_success,
        })

    if not verifier_enabled:
        tracker.update_lean_stats(
            0,
            0,
            definitions_compiled=False,
            failed_theorems=[],
            formal_success=True,
        )
        event_logger.emit("lean_result", {
            "skipped": True,
            "formal_success": True,
        })

    if not run_non_lean:
        metrics_filename = f"metrics_iter_{ctx.iteration}.json"
        metrics_path = os.path.join(ctx.workspace, metrics_filename)
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(tracker.get_summary(), f, indent=2)
        event_logger.emit("lean_only_result", {
            "formal_success": formal_success,
            "metrics_path": metrics_path,
        })
        if not formal_success:
            record_formal_feedback(ctx, formal_failures)
        return formal_success

    # Step 5: Simulation & Fuzzing
    print("Building Simulation Environment...")
    env = agents['architect']
    sim = tools['simulator']

    interface = env.extract_interface_description(ctx.solidity_code)
    semantic_context = "\n".join(ctx.semantic_guidance_texts) if ctx.semantic_guidance_texts else ""
    interface_for_simulation = interface
    if semantic_context:
        interface_for_simulation = f"{interface}\n\n[SEMANTIC OBLIGATIONS]\n{semantic_context}"
    mocks = ""
    mocks_feedback = ""
    max_retries = 3
    for attempt in range(max_retries):
        mocks = env.generate_mocks(ctx.solidity_code, error_feedback=mocks_feedback)
        sim.initialize_environment(ctx.solidity_code, mocks)
        success, log = sim.check_compilation()
        if success:
            print("      Mocks/Target compiled.")
            break
        print(f"      Mocks/Target compilation failed (Attempt {attempt+1}). Retrying...")
        mocks_feedback = log

    # === [LOOP A] Safety Rules Generation ===
    print("   Generating Safety Rules...")
    safety = ""
    check_logic = ""
    safety_feedback = ""

    for attempt in range(max_retries):
        safety, check_logic = env.generate_safety_rules(ctx.solidity_code, ctx.properties, error_feedback=safety_feedback)

        if not safety: # Parsing failed.
            safety_feedback = "Failed to parse JSON output."
            continue

        # Write the generated file and compile it.
        sim.save_file("src/SafetyRules.sol", safety)
        success, log = sim.check_compilation()

        if success:
            print("      SafetyRules compiled.")
            break
        else:
            print(f"      SafetyRules compilation failed (Attempt {attempt+1}). Retrying...")
            safety_feedback = log # Pass compiler errors to the next attempt.

    # === [LOOP B] Deployment Script Generation ===
    deploy = ""
    regex = {}
    deploy_feedback = ""

    if attacker_enabled:
        print("   Generating Deployment Script...")
        for attempt in range(max_retries):
            deploy, regex = env.generate_deploy_script(ctx.solidity_code, mocks, safety, error_feedback=deploy_feedback)

            if not deploy:
                deploy_feedback = "Failed to parse JSON output."
                continue

            # Write the revised file and compile it.
            sim.save_file("script/Deploy.s.sol", deploy)
            success, log = sim.check_compilation()

            if success:
                print("      Deploy Script compiled.")
                break
            else:
                print(f"      Deploy Script compilation failed (Attempt {attempt+1}). Retrying...")
                deploy_feedback = log
    else:
        print("   SKIPPING Deployment Script (Attacker disabled by ablation).")

    # === [LOOP C] Fuzz Test Generation ===
    print("   Generating Fuzz Tests...")
    fuzz_code = ""
    fuzz_feedback = ""
    is_fuzz_ready = False

    for attempt in range(max_retries):
        previous_fuzz_code = fuzz_code
        fuzz_code = env.generate_fuzz_test(
            ctx.solidity_code,
            safety,
            error_feedback=fuzz_feedback,
            previous_code=previous_fuzz_code,
            semantic_context=semantic_context,
        )

        # Use forge test for the compilation guard because forge build --skip test
        # may omit *.t.sol. Foundry must compile Invariant.t.sol before execution.
        sim.save_file("src/Invariant.t.sol", fuzz_code)
        success, log = sim.check_fuzz_compilation(fuzz_code)

        if success:
            print("      Fuzz Tests compiled.")
            is_fuzz_ready = True
            break
        else:
            print(f"      Fuzz Tests compilation failed (Attempt {attempt+1}). Retrying...")
            event_logger.emit("foundry_fuzz_harness_repair", {
                "phase": "preflight",
                "attempt": attempt + 1,
                "max_attempts": max_retries,
                "compile_pass": False,
                "error_tail": log[-3000:],
            })
            fuzz_feedback = log

    if not is_fuzz_ready:
        print("   Fuzzing skipped due to compilation errors.")
        append_unique(ctx.feedback_constraints, "Fix generated Foundry invariant harness or target incompatibility; fuzzing must compile and run.")

    # === [LOOP D] Semantic Probe Generation (diagnostic, separate from paper Foundry metrics) ===
    semantic_probe_enabled = (
        attacker_enabled
        and bool(ctx.semantic_obligations)
        and os.getenv("LEVER_SEMANTIC_PROBE", "1").lower() not in {"0", "false", "no"}
    )
    semantic_probe_code = ""
    semantic_probe_feedback = ""
    is_semantic_probe_ready = False

    if semantic_probe_enabled:
        print("   Generating Semantic Probe Tests...")
        for attempt in range(max_retries):
            previous_semantic_probe_code = semantic_probe_code
            semantic_probe_code = env.generate_semantic_probe_test(
                ctx.solidity_code,
                safety,
                semantic_context=semantic_context,
                error_feedback=semantic_probe_feedback,
                previous_code=previous_semantic_probe_code,
            )
            sim.save_file("src/SemanticProbe.t.sol", semantic_probe_code)
            success, log = sim.check_semantic_probe_compilation(semantic_probe_code)

            if success:
                print("      Semantic Probe Tests compiled.")
                is_semantic_probe_ready = True
                break

            print(f"      Semantic Probe compilation failed (Attempt {attempt+1}). Retrying...")
            event_logger.emit("semantic_probe_harness_repair", {
                "phase": "preflight",
                "attempt": attempt + 1,
                "max_attempts": max_retries,
                "compile_pass": False,
                "error_tail": log[-3000:],
            })
            semantic_probe_feedback = log

        if not is_semantic_probe_ready:
            print("   Semantic probe skipped due to compilation errors.")
    else:
        print("   Semantic Probe skipped (disabled, no attacker, or no semantic obligations).")

    # === EXECUTION PHASE ===
    # All generated files passed compilation guards; begin the simulations.

    full_execution_log = []
    full_execution_log.append(f"=== [TIMESTAMP] {time.strftime('%Y-%m-%d %H:%M:%S')} ===")

    # ---------------------------------------------------------
    # Mode 1: agent simulation for logic-level attacks.
    # ---------------------------------------------------------
    if attacker_enabled:
        print("   [Mode 1] Agent-Based Simulation...")

        # Execute the generated attack script.
        try:
            attack_result = sim.run_fuzz_and_attack(
                target_sol=ctx.solidity_code,
                mocks_sol=mocks,
                safety_sol=safety,
                deploy_script=deploy,
                regex_config=regex,
                interface_desc=interface_for_simulation,
                safety_check_logic=check_logic
            )
        finally:
            if hasattr(sim, "close"):
                sim.close()
        is_safe_ai = attack_result.get("safe", False)
        logs_ai = attack_result.get("log", "")
    else:
        print("   [Mode 1] Agent-Based Simulation skipped (no_attacker ablation).")
        attack_result = {
            "safe": True,
            "log": "SKIPPED_NO_ATTACKER_ABLATION",
            "attack_episodes": 0,
            "breaches": 0,
            "traces": [],
            "infra_broken": False,
        }
        is_safe_ai, logs_ai = True, attack_result["log"]

    # Normalize the simulation log format.
    ai_log_str = logs_ai if isinstance(logs_ai, str) else "\n".join(logs_ai)
    full_execution_log.append(f"\n=== AGENT SIMULATION LOGS ===\n{ai_log_str}")

    # Record whether the agent found a successful attack.
    tracker.update_from_agent_log(ai_log_str)
    tracker.update_attack_result(attack_result)
    event_logger.emit("agent_simulation_result", {
        "safe": attack_result.get("safe"),
        "attack_episodes": attack_result.get("attack_episodes", 0),
        "breaches": attack_result.get("breaches", 0),
        "infra_broken": attack_result.get("infra_broken", False),
        "deployment_success": attack_result.get("deployment_success", False),
        "address_parse_success": attack_result.get("address_parse_success", False),
        "front_run_attempts": attack_result.get("front_run_attempts", 0),
        "back_run_attempts": attack_result.get("back_run_attempts", 0),
        "trace_count": len(attack_result.get("traces", [])),
        "foundry_summary": attack_result.get("foundry_summary", {}),
    })

    restore_simulation_sources_for_dynamic_modes(
        sim,
        ctx.solidity_code,
        mocks,
        safety,
        deploy_code=deploy,
        fuzz_code=fuzz_code if is_fuzz_ready else "",
        semantic_probe_code=semantic_probe_code if is_semantic_probe_ready else "",
    )
    event_logger.emit("simulation_sources_restored_after_agent", {
        "has_fuzz": bool(is_fuzz_ready),
        "has_semantic_probe": bool(is_semantic_probe_ready),
    })

    if not is_safe_ai:
        print("   Agent simulation reported a potential vulnerability.")

    # ---------------------------------------------------------
    # Mode 2: Foundry fuzzing and invariant checks.
    # ---------------------------------------------------------
    is_safe_fuzz = is_fuzz_ready
    logs_fuzz = "Skipped"

    if is_fuzz_ready:
        print("   [Mode 2] Running Foundry Fuzzing...")

        # Execute the fuzzing stage.
        foundry_result = sim.run_foundry_fuzzing(fuzz_code)
        is_safe_fuzz = foundry_result.get("safe", False)
        logs_fuzz = foundry_result.get("log", "")
        if not foundry_result.get("foundry_compile_pass", False):
            print("   Foundry fuzz harness failed during test run. Attempting harness repair...")
            repair_feedback = logs_fuzz
            for repair_attempt in range(max_retries):
                previous_fuzz_code = fuzz_code
                fuzz_code = env.generate_fuzz_test(
                    ctx.solidity_code,
                    safety,
                    error_feedback=repair_feedback,
                    previous_code=previous_fuzz_code,
                    semantic_context=semantic_context,
                )
                sim.save_file("src/Invariant.t.sol", fuzz_code)
                compile_ok, compile_log = sim.check_fuzz_compilation(fuzz_code)
                event_logger.emit("foundry_fuzz_harness_repair", {
                    "phase": "runtime",
                    "attempt": repair_attempt + 1,
                    "max_attempts": max_retries,
                    "compile_pass": compile_ok,
                    "error_tail": "" if compile_ok else compile_log[-3000:],
                })
                if not compile_ok:
                    repair_feedback = compile_log
                    continue

                foundry_result = sim.run_foundry_fuzzing(fuzz_code)
                is_safe_fuzz = foundry_result.get("safe", False)
                logs_fuzz = foundry_result.get("log", "")
                if foundry_result.get("foundry_compile_pass", False):
                    break
                repair_feedback = logs_fuzz

            if not foundry_result.get("foundry_compile_pass", False):
                is_safe_fuzz = False
                append_unique(ctx.feedback_constraints, "Fix generated Foundry invariant harness or target incompatibility; fuzzing must compile and run.")
        full_execution_log.append(f"\n=== FOUNDRY FUZZING RAW OUTPUT ===\n{logs_fuzz}")

        # Record invariant outcomes and call counts.
        tracker.update_from_fuzz_log(logs_fuzz)
        tracker.update_foundry_result(foundry_result)
        event_logger.emit("foundry_fuzz_result", {
            "safe": foundry_result.get("safe"),
            "foundry_compile_pass": foundry_result.get("foundry_compile_pass"),
            "foundry_test_pass": foundry_result.get("foundry_test_pass"),
            "infra_broken": foundry_result.get("infra_broken", False),
            "broken_invariants": foundry_result.get("broken_invariants", []),
            "fuzz_runs": foundry_result.get("fuzz_runs", 0),
            "foundry_summary": foundry_result.get("foundry_summary", {}),
        })
    else:
        skipped_foundry = {
            "safe": False,
            "foundry_compile_pass": False,
            "foundry_test_pass": False,
            "infra_broken": True,
            "broken_invariants": [],
            "fuzz_runs": 0,
        }
        tracker.update_foundry_result(skipped_foundry)
        event_logger.emit("foundry_fuzz_result", {**skipped_foundry, "skipped": True})

    # ---------------------------------------------------------
    # Diagnostic Semantic Probe (kept separate from paper Foundry metrics)
    # ---------------------------------------------------------
    semantic_probe_blocks_success = False
    semantic_probe_needs_review = False
    semantic_probe_pending_constraint = ""
    logs_semantic_probe = "Skipped"
    semantic_probe_result = {
        "available": False,
        "safe": None,
        "semantic_probe_compile_pass": None,
        "semantic_probe_test_pass": None,
        "infra_broken": False,
        "semantic_failures": [],
        "foundry_summary": {},
    }

    if semantic_probe_enabled and is_semantic_probe_ready:
        semantic_probe_result = sim.run_semantic_probe(semantic_probe_code)
        logs_semantic_probe = semantic_probe_result.get("log", "")

        if not semantic_probe_result.get("semantic_probe_compile_pass", False):
            print("   Semantic probe failed during test run. Attempting probe repair...")
            repair_feedback = logs_semantic_probe
            for repair_attempt in range(max_retries):
                previous_semantic_probe_code = semantic_probe_code
                semantic_probe_code = env.generate_semantic_probe_test(
                    ctx.solidity_code,
                    safety,
                    semantic_context=semantic_context,
                    error_feedback=repair_feedback,
                    previous_code=previous_semantic_probe_code,
                )
                sim.save_file("src/SemanticProbe.t.sol", semantic_probe_code)
                compile_ok, compile_log = sim.check_semantic_probe_compilation(semantic_probe_code)
                event_logger.emit("semantic_probe_harness_repair", {
                    "phase": "runtime",
                    "attempt": repair_attempt + 1,
                    "max_attempts": max_retries,
                    "compile_pass": compile_ok,
                    "error_tail": "" if compile_ok else compile_log[-3000:],
                })
                if not compile_ok:
                    repair_feedback = compile_log
                    continue

                semantic_probe_result = sim.run_semantic_probe(semantic_probe_code)
                logs_semantic_probe = semantic_probe_result.get("log", "")
                if semantic_probe_result.get("semantic_probe_compile_pass", False):
                    break
                repair_feedback = logs_semantic_probe

        if (
            semantic_probe_result.get("semantic_probe_test_pass") is False
            and not semantic_probe_result.get("infra_broken", False)
        ):
            semantic_probe_needs_review = True
            semantic_probe_blocks_success = (
                os.getenv("LEVER_SEMANTIC_PROBE_FORCE_ITERATES", "1").lower()
                not in {"0", "false", "no"}
            )
            semantic_probe_pending_constraint = (
                "Address the requirement-level semantic probe failure with a general contract-level fix. "
                "Do not patch for one sample; preserve positive lifecycle behavior and enforce the inferred business rule. "
                f"Probe evidence tail: {str(logs_semantic_probe)[-1600:]}"
            )

        full_execution_log.append(f"\n=== SEMANTIC PROBE RAW OUTPUT ===\n{logs_semantic_probe}")
        tracker.update_semantic_probe_result(semantic_probe_result, enabled=True)
        event_logger.emit("semantic_probe_result", {
            "safe": semantic_probe_result.get("safe"),
            "semantic_probe_compile_pass": semantic_probe_result.get("semantic_probe_compile_pass"),
            "semantic_probe_test_pass": semantic_probe_result.get("semantic_probe_test_pass"),
            "infra_broken": semantic_probe_result.get("infra_broken", False),
            "semantic_failures": semantic_probe_result.get("semantic_failures", []),
            "needs_analyst_review": semantic_probe_needs_review,
            "blocks_iteration": semantic_probe_blocks_success,
            "pending_analyst_confirmation": bool(semantic_probe_pending_constraint),
            "foundry_summary": semantic_probe_result.get("foundry_summary", {}),
            "metric_policy": "diagnostic_only_not_paper_foundry_metric",
        })
    else:
        if semantic_probe_enabled:
            semantic_probe_result.update({
                "available": False,
                "semantic_probe_compile_pass": False,
                "semantic_probe_test_pass": None,
                "infra_broken": True,
            })
        full_execution_log.append(f"\n=== SEMANTIC PROBE RAW OUTPUT ===\n{logs_semantic_probe}")
        tracker.update_semantic_probe_result(semantic_probe_result, enabled=semantic_probe_enabled)
        event_logger.emit("semantic_probe_result", {
            **semantic_probe_result,
            "skipped": True,
            "metric_policy": "diagnostic_only_not_paper_foundry_metric",
        })

    tracker.update_dynamic_results(agent_safe=is_safe_ai, fuzzing_safe=is_safe_fuzz)

    # ---------------------------------------------------------
    # Standard Correctness / External Audit Metrics
    # ---------------------------------------------------------
    if run_slither:
        print("   Running Slither standard-correctness check...")
        target_path = os.path.join(sim.src_dir, "Target.sol")
        slither_result = run_slither_check(target_path)
        tracker.update_slither_result(slither_result, enabled=True)
    else:
        tracker.update_slither_result({"slither_pass": None, "infra_broken": False}, enabled=False)

    if run_llm_audit:
        print("   Running LLM audit pass-rate check...")
        audit_result = agents["auditor"].audit_contract(ctx.solidity_code, ctx.properties)
        tracker.update_llm_audit_result(audit_result, enabled=True)
        ctx.checkpoint.save_stage("llm_audit_result", audit_result)
    else:
        tracker.update_llm_audit_result({"audit_pass": None}, enabled=False)

    # ---------------------------------------------------------
    # Semantic Suspicion Analysis
    # ---------------------------------------------------------
    semantic_suspicion_blocks_success = False
    dynamic_clean = bool(is_safe_ai and is_safe_fuzz)
    dynamic_infra_clean = not (
        tracker.stats.get("attack_infra_broken")
        or tracker.stats.get("foundry_infra_broken")
    )
    if (
        attacker_enabled
        and dynamic_clean
        and dynamic_infra_clean
        and ctx.semantic_obligations
        and hasattr(agents.get("analyst"), "analyze_semantic_suspicion")
    ):
        foundry_structured_summary = {
            "foundry_compile_pass": tracker.stats.get("foundry_compile_pass"),
            "foundry_test_pass": tracker.stats.get("foundry_test_pass"),
            "broken_foundry_invariants": tracker.stats.get("broken_foundry_invariants", []),
            "foundry_fuzz_runs": tracker.stats.get("foundry_fuzz_runs", 0),
            "fuzz_calls": tracker.stats.get("fuzz_calls", 0),
            "foundry_summary": tracker.stats.get("foundry_summary", {}),
        }
        semantic_probe_structured_summary = {
            "semantic_probe_available": tracker.stats.get("semantic_probe_available"),
            "semantic_probe_compile_pass": tracker.stats.get("semantic_probe_compile_pass"),
            "semantic_probe_test_pass": tracker.stats.get("semantic_probe_test_pass"),
            "semantic_probe_infra_broken": tracker.stats.get("semantic_probe_infra_broken"),
            "semantic_probe_failures": tracker.stats.get("semantic_probe_failures", []),
            "semantic_probe_summary": tracker.stats.get("semantic_probe_summary", {}),
            "semantic_probe_needs_review": semantic_probe_needs_review,
            "metric_policy": "diagnostic_only_not_paper_foundry_metric",
        }
        semantic_evidence_snippets = extract_semantic_evidence_snippets(
            "\n".join([ai_log_str, str(logs_fuzz), str(logs_semantic_probe)]),
        )
        suspicion_logs = f"""
        [SELECTED SEMANTIC OBLIGATIONS]
        {json.dumps(ctx.semantic_obligations, indent=2, ensure_ascii=False)}

        [SELECTED PROPERTIES]
        {json.dumps(ctx.properties, indent=2, ensure_ascii=False)}

        [SAFETY RULES CODE]
        {safety[:8000]}

        [FOUNDRY STRUCTURED SUMMARY]
        {json.dumps(foundry_structured_summary, indent=2, ensure_ascii=False)}

        [SEMANTIC PROBE STRUCTURED SUMMARY]
        {json.dumps(semantic_probe_structured_summary, indent=2, ensure_ascii=False)}

        [SEMANTIC PROBE SOURCE]
        {semantic_probe_code[:12000] if semantic_probe_code else "None"}

        [SEMANTIC EVIDENCE SNIPPETS FROM FULL LOGS]
        {semantic_evidence_snippets or "None"}

        [AGENT LOGS]
        {ai_log_str[-12000:]}

        [FOUNDRY OUTPUT TAIL]
        {str(logs_fuzz)[-6000:]}

        [SEMANTIC PROBE OUTPUT TAIL]
        {str(logs_semantic_probe)[-6000:]}
        """
        suspicion_result = agents["analyst"].analyze_semantic_suspicion(suspicion_logs)
        tracker.update_semantic_suspicion(suspicion_result)
        ctx.checkpoint.save_stage("last_semantic_suspicion", suspicion_result)
        event_logger.emit("semantic_suspicion_analysis", suspicion_result)

        confidence = str(suspicion_result.get("confidence", "")).lower()
        suspicion_property = suspicion_result.get("property", {})
        if suspicion_result.get("suspicious") and confidence in {"medium", "high"} and suspicion_property:
            suspicion_record = {
                "iteration": ctx.iteration,
                "analysis": suspicion_result.get("analysis", ""),
                "constraint": suspicion_result.get("constraint", ""),
                "property": suspicion_property,
                "source": "semantic_suspicion",
            }
            property_text = attack_property_to_text(suspicion_record)
            if property_text and property_text not in [attack_property_to_text(r) for r in ctx.attack_properties]:
                ctx.attack_properties.append(suspicion_record)
                ctx.security_patches.append(property_text)
                ctx.checkpoint.save_stage("attack_properties", ctx.attack_properties)
                tracker.stats["attack_derived_properties"] = ctx.attack_properties
                print(f"   Suspicious semantic evidence found: {property_text[:90]}...")

            constraint = suspicion_result.get("constraint", "")
            append_unique(ctx.feedback_constraints, constraint)
            if semantic_probe_pending_constraint:
                append_unique(ctx.feedback_constraints, semantic_probe_pending_constraint)
            semantic_suspicion_blocks_success = os.getenv("LEVER_SUSPICION_ITERATES", "1").lower() not in {"0", "false", "no"}
        elif suspicion_result.get("suspicious"):
            print("   Low-confidence semantic suspicion recorded, but it will not force iteration.")

    # ---------------------------------------------------------
    # Result Processing: Saving Logs & Metrics
    # ---------------------------------------------------------

    # 1. Save the detailed raw log.
    log_filename = f"sim_detail_iter_{ctx.iteration}.log"
    log_path = os.path.join(ctx.workspace, log_filename)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(full_execution_log))
    print(f"   Detailed Logs saved to: {log_filename}")

    # 3. Save quantitative metrics as JSON.
    metrics_filename = f"metrics_iter_{ctx.iteration}.json"
    metrics_path = os.path.join(ctx.workspace, metrics_filename)
    tracker.stats["attack_derived_properties"] = ctx.attack_properties
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(tracker.get_summary(), f, indent=2)
    print(f"   📊 Metrics saved to: {metrics_filename}")

    structured_summary = {
        "iteration": ctx.iteration,
        "metrics_path": metrics_path,
        "detail_log_path": log_path,
        "events_path": event_logger.path,
        "command_log_dir": os.path.join(sim.root_dir, "command_logs"),
        "metrics": tracker.get_summary(),
        "command_count": len(getattr(sim, "command_history", [])),
    }
    summary_filename = f"summary_iter_{ctx.iteration}.json"
    summary_path = os.path.join(ctx.workspace, summary_filename)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(structured_summary, f, indent=2, ensure_ascii=False)
    event_logger.emit("iteration_summary_written", {
        "summary_path": summary_path,
        "metrics_path": metrics_path,
        "detail_log_path": log_path,
        "command_log_dir": structured_summary["command_log_dir"],
        "command_count": structured_summary["command_count"],
    })
    print(f"   Structured summary saved to: {summary_filename}")

    # ---------------------------------------------------------
    # Final Analysis & Feedback Loop
    # ---------------------------------------------------------
    agent_verified_or_unavailable = (
        (not attacker_enabled)
        or bool(is_safe_ai)
        or (
            bool(tracker.stats.get("attack_infra_broken"))
            and int(tracker.stats.get("attack_breaches") or 0) == 0
        )
    )
    formal_gate_ok = formal_success or not formal_blocks_dynamic
    is_safe = (
        formal_gate_ok
        and agent_verified_or_unavailable
        and is_safe_fuzz
        and not semantic_suspicion_blocks_success
        and not semantic_probe_blocks_success
    )

    if is_safe:
        if attacker_enabled and tracker.stats.get("attack_infra_broken") and not is_safe_ai:
            print("   Agent simulation unavailable; accepted under the original core policy because no breach was observed.")
        print("\nPIPELINE CRITERIA PASSED under legacy_core_lenient policy.")
        return True
    else:
        if not formal_success:
            if formal_blocks_dynamic:
                print("   Formal verification did not prove all selected properties.")
            else:
                print("   Formal verification incomplete; recorded separately and not blocking dynamic result.")

        dynamic_failure = (
            (attacker_enabled and not is_safe_ai and not tracker.stats.get("attack_infra_broken"))
            or (not is_safe_fuzz)
        )
        if (semantic_suspicion_blocks_success or semantic_probe_blocks_success) and not dynamic_failure:
            if semantic_probe_blocks_success:
                print("   Semantic probe evidence is strong enough to trigger another iteration.")
            if semantic_suspicion_blocks_success:
                print("   Semantic suspicion is strong enough to trigger another iteration.")
            return False

        if dynamic_failure:
            if os.getenv("LEVER_SKIP_FAILURE_ANALYSIS", "0").lower() in {"1", "true", "yes"}:
                print("   Skipping failure analysis/repair feedback by LEVER_SKIP_FAILURE_ANALYSIS.")
                return False

            print("   Local diagnostic failure detected. Analyzing for feedback...")

            # Combine stage logs for analyst review.
            foundry_structured_summary = {
                "foundry_compile_pass": tracker.stats.get("foundry_compile_pass"),
                "foundry_test_pass": tracker.stats.get("foundry_test_pass"),
                "broken_foundry_invariants": tracker.stats.get("broken_foundry_invariants", []),
                "foundry_fuzz_runs": tracker.stats.get("foundry_fuzz_runs", 0),
                "fuzz_calls": tracker.stats.get("fuzz_calls", 0),
                "foundry_summary": tracker.stats.get("foundry_summary", {}),
                "semantic_probe_diagnostic": {
                    "semantic_probe_available": tracker.stats.get("semantic_probe_available"),
                    "semantic_probe_compile_pass": tracker.stats.get("semantic_probe_compile_pass"),
                    "semantic_probe_test_pass": tracker.stats.get("semantic_probe_test_pass"),
                    "semantic_probe_infra_broken": tracker.stats.get("semantic_probe_infra_broken"),
                    "semantic_probe_failures": tracker.stats.get("semantic_probe_failures", []),
                    "metric_policy": "diagnostic_only_not_paper_foundry_metric",
                },
            }
            semantic_evidence_snippets = extract_semantic_evidence_snippets(
                "\n".join([ai_log_str, str(logs_fuzz), str(logs_semantic_probe)]),
            )
            combined_logs = f"""
            [SELECTED SEMANTIC OBLIGATIONS]
            {json.dumps(ctx.semantic_obligations, indent=2, ensure_ascii=False)}

            [SELECTED PROPERTIES]
            {json.dumps(ctx.properties, indent=2, ensure_ascii=False)}

            [FOUNDRY STRUCTURED SUMMARY]
            {json.dumps(foundry_structured_summary, indent=2, ensure_ascii=False)}

            [SEMANTIC EVIDENCE SNIPPETS FROM FULL LOGS]
            {semantic_evidence_snippets or "None"}

            [AGENT LOGS]
            {ai_log_str[-12000:]}

            [FUZZING FAILURE]
            {logs_fuzz[-5000:]}

            [SEMANTIC PROBE DIAGNOSTIC TAIL]
            {str(logs_semantic_probe)[-5000:]}
            """

            structured_analysis = agents['analyst'].analyze_attack_structured(combined_logs)
            ctx.checkpoint.save_stage("last_attack_analysis", structured_analysis)

            attack_property = structured_analysis.get("property", {})
            property_status = str(attack_property.get("status", "")).lower() if attack_property else ""
            if property_status == "harness_issue":
                print("   Analyst classified this as a harness/infrastructure issue; target code repair is not triggered.")
                tracker.stats["harness_issue"] = True
                tracker.stats["harness_issue_analysis"] = structured_analysis
                harness_issue_record = {
                    "iteration": ctx.iteration,
                    "analysis": structured_analysis,
                }
                ctx.checkpoint.save_stage("last_harness_issue", harness_issue_record)
                event_logger.emit("harness_issue_classified", {
                    "analysis": structured_analysis.get("analysis", ""),
                    "constraint": structured_analysis.get("constraint", ""),
                    "property": attack_property,
                })
                with open(metrics_path, "w", encoding="utf-8") as f:
                    json.dump(tracker.get_summary(), f, indent=2)

                foundry_clean = (
                    tracker.stats.get("foundry_compile_pass") is True
                    and tracker.stats.get("foundry_test_pass") is True
                    and not tracker.stats.get("broken_foundry_invariants", [])
                )
                no_attack_breach = tracker.stats.get("attack_breaches", 0) == 0
                if foundry_clean and no_attack_breach:
                    print("   No Foundry failure or concrete attack breach was observed; accepted under the lenient policy despite the harness issue.")
                    return True
                return False

            if attack_property:
                attack_record = {
                    "iteration": ctx.iteration,
                    "analysis": structured_analysis.get("analysis", ""),
                    "constraint": structured_analysis.get("constraint", ""),
                    "property": attack_property,
                }
                property_text = attack_property_to_text(attack_record)
                if property_text and property_text not in [attack_property_to_text(r) for r in ctx.attack_properties]:
                    ctx.attack_properties.append(attack_record)
                    ctx.security_patches.append(property_text)
                    ctx.checkpoint.save_stage("attack_properties", ctx.attack_properties)
                    tracker.stats["attack_derived_properties"] = ctx.attack_properties
                    with open(metrics_path, "w", encoding="utf-8") as f:
                        json.dump(tracker.get_summary(), f, indent=2)
                    print(f"   New Attack-Derived Property: {property_text[:80]}...")

            constraint = structured_analysis.get("constraint", "")
            append_unique(ctx.feedback_constraints, constraint)

            if not attack_property and not constraint:
                analysis = agents['analyst'].analyze_attack(combined_logs)
                if "[PROPERTY]" in analysis:
                    new_prop = analysis.split("[PROPERTY]")[1].strip()
                    if new_prop not in ctx.security_patches:
                        ctx.security_patches.append(new_prop)
                        print(f"   New Security Patch: {new_prop[:50]}...")

                if "[CONSTRAINT]" in analysis:
                    con = analysis.split("[CONSTRAINT]")[1].split("[")[0].strip()
                    append_unique(ctx.feedback_constraints, con)

        return False
