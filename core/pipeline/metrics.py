import re
from typing import Dict

class MetricsTracker:
    def __init__(self):
        self.stats = {
            "ablation_mode": "full",
            "verifier_enabled": True,
            "attacker_enabled": True,
            "compilation_success": False,
            "total_invariants": 0,      # Invariants defined
            "passed_invariants": 0,     # Fuzzer passed
            "broken_invariants": 0,     # Fuzzer broken
            "agent_successes": 0,       # Agent exploits
            "fuzz_calls": 0,            # Total fuzz calls
            "unique_errors": set(),
            "infra_broken": False,
            # [Added] Formal Verification Stats
            "lean_theorems_total": 0,
            "lean_theorems_proven": 0,
            "verification_rate": None,
            "avg_verified_properties": 0,
            "lean_definitions_compiled": False,
            "lean_failed_theorems": [],
            "formal_success": False,
            "agent_simulation_safe": None,
            "attack_episodes": 0,
            "attack_breaches": 0,
            "attack_success_rate": None,
            "attack_traces": [],
            "attack_infra_broken": False,
            "attack_rounds": 0,
            "attack_scheduler": "",
            "front_run_attempts": 0,
            "back_run_attempts": 0,
            "lifecycle_setup": {},
            "honest_intents": [],
            "schedule_phases": [],
            "attack_derived_properties": [],
            "fuzzing_safe": None,
            "foundry_compile_pass": None,
            "foundry_test_pass": None,
            "foundry_infra_broken": False,
            "broken_foundry_invariants": [],
            "foundry_fuzz_runs": 0,
            "foundry_summary": {},
            "llm_audit_enabled": True,
            "llm_audit_pass": None,
            "llm_audit_high": None,
            "llm_audit_medium": None,
            "llm_audit_low": None,
            "llm_audit_findings": [],
            "llm_audit_infra_broken": False,
            "slither_enabled": True,
            "slither_pass": None,
            "slither_high": None,
            "slither_medium": None,
            "slither_low": None,
            "slither_infra_broken": False,
            "slither_error": "",
            "semantic_obligations_enabled": True,
            "semantic_obligation_count": 0,
            "semantic_obligations": [],
            "semantic_property_texts": [],
            "semantic_guidance_texts": [],
            "semantic_confidence_note": "",
            "semantic_suspicion": False,
            "semantic_suspicion_confidence": "",
            "semantic_suspicion_analysis": {},
            "semantic_suspicion_properties": [],
            "semantic_probe_enabled": False,
            "semantic_probe_available": False,
            "semantic_probe_safe": None,
            "semantic_probe_compile_pass": None,
            "semantic_probe_test_pass": None,
            "semantic_probe_infra_broken": False,
            "semantic_probe_failures": [],
            "semantic_probe_summary": {},
            "semantic_probe_confidence_note": (
                "Semantic probes are requirement-derived diagnostics and are not counted as paper Foundry fuzz metrics."
            ),
        }

    def set_mode_flags(self, ablation_mode: str, verifier_enabled: bool, attacker_enabled: bool):
        self.stats["ablation_mode"] = ablation_mode
        self.stats["verifier_enabled"] = verifier_enabled
        self.stats["attacker_enabled"] = attacker_enabled

    def update_compilation(self, success: bool):
        self.stats["compilation_success"] = success

    def update_from_fuzz_log(self, log_content):
        # 1. Invariant Stats
        self.stats["passed_invariants"] += len(re.findall(r"\[PASS\]", log_content))
        broken = re.findall(r"\[FAIL", log_content)
        self.stats["broken_invariants"] += len(broken)
        self.stats["total_invariants"] = self.stats["passed_invariants"] + self.stats["broken_invariants"]

        # 2. Call Stats
        try:
            table_rows = re.findall(r"\|\s*Handler\s*\|\s*\w+\s*\|\s*(\d+)\s*\|\s*(\d+)", log_content)
            for calls, reverts in table_rows:
                self.stats["fuzz_calls"] += int(calls)
        except:
            pass

    def update_from_agent_log(self, log_content):
        if "VULNERABILITY:" in log_content or "SUCCESS:" in log_content or "CRITICAL:" in log_content:
            self.stats["agent_successes"] = 1

    def update_lean_stats(self, total, proven, definitions_compiled=False, failed_theorems=None, formal_success=False):
        self.stats["lean_theorems_total"] = total
        self.stats["lean_theorems_proven"] = proven
        self.stats["verification_rate"] = (proven / total) if total else None
        self.stats["avg_verified_properties"] = proven
        self.stats["lean_definitions_compiled"] = definitions_compiled
        self.stats["lean_failed_theorems"] = failed_theorems or []
        self.stats["formal_success"] = formal_success

    def update_dynamic_results(self, agent_safe=None, fuzzing_safe=None):
        self.stats["agent_simulation_safe"] = agent_safe
        self.stats["fuzzing_safe"] = fuzzing_safe

    def update_attack_result(self, result: Dict):
        self.stats["agent_simulation_safe"] = result.get("safe")
        self.stats["attack_episodes"] = result.get("attack_episodes", 0)
        self.stats["attack_breaches"] = result.get("breaches", 0)
        episodes = self.stats["attack_episodes"]
        self.stats["attack_success_rate"] = (self.stats["attack_breaches"] / episodes) if episodes else None
        self.stats["attack_traces"] = result.get("traces", [])
        self.stats["attack_infra_broken"] = result.get("infra_broken", False)
        self.stats["attack_rounds"] = result.get("total_rounds", 0)
        self.stats["attack_scheduler"] = result.get("scheduler", "")
        self.stats["front_run_attempts"] = result.get("front_run_attempts", 0)
        self.stats["back_run_attempts"] = result.get("back_run_attempts", 0)
        self.stats["lifecycle_setup"] = result.get("lifecycle_setup", {})
        self.stats["honest_intents"] = result.get("honest_intents", [])
        self.stats["schedule_phases"] = result.get("schedule_phases", [])
        self.stats["infra_broken"] = self.stats["infra_broken"] or self.stats["attack_infra_broken"]

    def update_foundry_result(self, result: Dict):
        self.stats["fuzzing_safe"] = result.get("safe")
        self.stats["foundry_compile_pass"] = result.get("foundry_compile_pass")
        self.stats["foundry_test_pass"] = result.get("foundry_test_pass")
        self.stats["foundry_infra_broken"] = result.get("infra_broken", False)
        self.stats["broken_foundry_invariants"] = result.get("broken_invariants", [])
        self.stats["foundry_fuzz_runs"] = result.get("fuzz_runs", 0)
        self.stats["foundry_summary"] = result.get("foundry_summary", {})
        self.stats["compilation_success"] = bool(self.stats["foundry_compile_pass"])

        foundry_summary = self.stats["foundry_summary"] or {}
        suite_result = foundry_summary.get("suite_result") or {}
        run_result = foundry_summary.get("run_result") or {}
        passed_count = foundry_summary.get("passed_test_count")
        failed_count = foundry_summary.get("failed_test_count")
        if passed_count is None:
            passed_count = suite_result.get("passed", run_result.get("passed", 0))
        if failed_count is None:
            failed_count = suite_result.get("failed", run_result.get("failed", 0))
        self.stats["passed_invariants"] = int(passed_count or 0)
        self.stats["broken_invariants"] = int(failed_count or 0)
        self.stats["total_invariants"] = self.stats["passed_invariants"] + self.stats["broken_invariants"]
        self.stats["fuzz_calls"] = int(foundry_summary.get("fuzz_calls") or self.stats["fuzz_calls"] or 0)
        self.stats["infra_broken"] = self.stats["infra_broken"] or self.stats["foundry_infra_broken"]

    def update_llm_audit_result(self, result: Dict, enabled: bool = True):
        self.stats["llm_audit_enabled"] = enabled
        self.stats["llm_audit_pass"] = result.get("audit_pass")
        self.stats["llm_audit_high"] = result.get("high_severity_count")
        self.stats["llm_audit_medium"] = result.get("medium_severity_count")
        self.stats["llm_audit_low"] = result.get("low_severity_count")
        self.stats["llm_audit_findings"] = result.get("findings", [])
        self.stats["llm_audit_infra_broken"] = enabled and result.get("audit_pass") is None
        self.stats["infra_broken"] = self.stats["infra_broken"] or self.stats["llm_audit_infra_broken"]

    def update_slither_result(self, result: Dict, enabled: bool = True):
        self.stats["slither_enabled"] = enabled
        self.stats["slither_pass"] = result.get("slither_pass")
        self.stats["slither_high"] = result.get("high")
        self.stats["slither_medium"] = result.get("medium")
        self.stats["slither_low"] = result.get("low")
        self.stats["slither_error"] = result.get("error", "")
        self.stats["slither_infra_broken"] = result.get("infra_broken", False)
        self.stats["infra_broken"] = self.stats["infra_broken"] or self.stats["slither_infra_broken"]

    def update_semantic_suspicion(self, result: Dict):
        suspicious = bool(result.get("suspicious"))
        self.stats["semantic_suspicion"] = suspicious
        self.stats["semantic_suspicion_confidence"] = result.get("confidence", "")
        self.stats["semantic_suspicion_analysis"] = result
        if suspicious:
            prop = result.get("property", {})
            if prop:
                self.stats["semantic_suspicion_properties"].append(prop)

    def update_semantic_probe_result(self, result: Dict, enabled: bool = True):
        result = result or {}
        self.stats["semantic_probe_enabled"] = enabled
        self.stats["semantic_probe_available"] = bool(result.get("available"))
        self.stats["semantic_probe_safe"] = result.get("safe")
        self.stats["semantic_probe_compile_pass"] = result.get("semantic_probe_compile_pass")
        self.stats["semantic_probe_test_pass"] = result.get("semantic_probe_test_pass")
        self.stats["semantic_probe_infra_broken"] = bool(result.get("infra_broken", False))
        self.stats["semantic_probe_failures"] = result.get("semantic_failures", [])
        self.stats["semantic_probe_summary"] = result.get("foundry_summary", {})

    def get_summary(self):
        data = self.stats.copy()
        data["unique_errors"] = list(data["unique_errors"])
        return data
