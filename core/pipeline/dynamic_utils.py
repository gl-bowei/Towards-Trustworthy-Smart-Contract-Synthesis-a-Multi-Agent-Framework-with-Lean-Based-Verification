import os

def restore_simulation_sources_for_dynamic_modes(
    sim,
    target_code: str,
    mocks_code: str,
    safety_code: str,
    deploy_code: str = "",
    fuzz_code: str = "",
    semantic_probe_code: str = "",
):
    """Restore core files after stages that may reinitialize simulation_env."""
    os.makedirs(sim.src_dir, exist_ok=True)
    os.makedirs(sim.script_dir, exist_ok=True)
    sim.save_file("src/Target.sol", target_code)
    sim.save_file("src/Mocks.sol", mocks_code)
    sim.save_file("src/SafetyRules.sol", safety_code)
    if deploy_code:
        sim.save_file("script/Deploy.s.sol", deploy_code)
    if fuzz_code:
        sim.save_file("src/Invariant.t.sol", fuzz_code)
    if semantic_probe_code:
        sim.save_file("src/SemanticProbe.t.sol", semantic_probe_code)
