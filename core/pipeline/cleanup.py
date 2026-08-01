import os
import shutil

def cleanup_heavy_files(folder_path):
    """
    Remove heavy Foundry dependencies and build artifacts (lib, out, cache),
    retaining only core source files and logs.
    """
    if not os.path.exists(folder_path):
        return

    # Directory names that are safe to remove from managed workspaces.
    targets = ["lib", "out", "cache", "broadcast"]

    # Limit cleanup to simulation environments for additional safety.
    sim_env = os.path.join(folder_path, "simulation_env")
    if os.path.exists(sim_env):
        for target in targets:
            target_path = os.path.join(sim_env, target)
            if os.path.exists(target_path):
                try:
                    shutil.rmtree(target_path)
                    print(f"   🧹 Cleaned: {target_path}")
                except Exception as e:
                    print(f"   ⚠️ Failed to clean {target_path}: {e}")
