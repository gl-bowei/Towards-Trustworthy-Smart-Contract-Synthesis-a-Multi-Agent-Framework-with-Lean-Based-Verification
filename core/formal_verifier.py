import os
import re
import subprocess
from typing import List

class FormalVerifier:
    def __init__(self, project_root="lean_project", mode=None):
        # Use absolute path to avoid cwd confusion
        self.project_root = os.path.abspath(project_root)
        self.mode = (mode or os.getenv("LEVER_LEAN_MODE", "flat")).lower()
        if self.mode not in {"flat", "lake"}:
            self.mode = "flat"
        self.lean_version = os.getenv("LEVER_LEAN_VERSION", "leanprover/lean4:v4.12.0")
        self.source_dir = (
            self.project_root
            if self.mode == "flat"
            else os.path.join(self.project_root, "Verification")
        )
        self._ensure_environment()

    def _ensure_environment(self):
        """
        Sets up a Lean environment.
        Default is a flat directory compiled with `lean file.lean`, which avoids
        per-case Lake build caches. Set LEVER_LEAN_MODE=lake for the old layout.
        """
        if not os.path.exists(self.project_root):
            label = "Lean Project" if self.mode == "lake" else "Lean Workspace"
            print(f"   🛠️  Initializing {label} at {self.project_root}...")

        os.makedirs(self.project_root, exist_ok=True)
        os.makedirs(self.source_dir, exist_ok=True)

        toolchain_path = os.path.join(self.project_root, "lean-toolchain")
        if not os.path.exists(toolchain_path):
            with open(toolchain_path, "w") as f:
                f.write(self.lean_version)

        if self.mode == "lake":
            lakefile_path = os.path.join(self.project_root, "lakefile.lean")
            if not os.path.exists(lakefile_path):
                # Create lakefile.lean (Build configuration)
                lakefile_content = """
                import Lake
                open Lake DSL

                package verification {
                  -- basic configuration
                }

                lean_lib Verification {
                  -- default options
                }
                """
                with open(lakefile_path, "w") as f:
                    f.write(lakefile_content)

    def save_file(self, filename: str, content: str):
        """
        Saves a Lean file into the verifier workspace.
        """
        if os.path.isabs(filename) or ".." in filename.replace("\\", "/").split("/"):
            raise ValueError(f"Unsafe Lean filename: {filename!r}")
        if not filename.endswith(".lean"):
            raise ValueError(f"Lean verifier only accepts .lean files: {filename!r}")
        path = os.path.abspath(os.path.join(self.source_dir, filename))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(content)

    def verify_file(self, filename: str) -> List[str]:
        """
        Compiles a specific Lean file. Flat mode runs `lean file.lean`; Lake mode
        keeps the old `lake build Verification.Module` behavior.
        """
        if self.mode == "lake":
            module_name = "Verification." + filename.replace(".lean", "")
            cmd = ["lake", "build", module_name]
            cwd = self.project_root
            missing_tool_msg = "System Error: 'lake' command not found. Is Lean 4 installed?"
        else:
            file_path = os.path.abspath(os.path.join(self.source_dir, filename))
            cmd = ["lean", file_path]
            cwd = self.project_root
            missing_tool_msg = "System Error: 'lean' command not found. Is Lean 4 installed?"

        try:
            result = subprocess.run(
                cmd,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=int(os.getenv("LEVER_LEAN_TIMEOUT", "120")),
            )
        except FileNotFoundError:
            return [missing_tool_msg]
        except subprocess.TimeoutExpired:
            return [f"Lean verification timed out after {os.getenv('LEVER_LEAN_TIMEOUT', '120')} seconds."]

        if result.returncode == 0:
            return []
        return self._parse_errors(result.stdout + result.stderr)

    def _parse_errors(self, output: str) -> List[str]:
        """
        Parses compiler output for specific error messages.
        """
        errors = []
        lines = output.split('\n')

        for line in lines:
            if "error:" in line:
                parts = line.split("error:")
                if len(parts) > 1:
                    clean_msg = parts[1].strip()
                    location = parts[0].strip()
                    location_match = None
                    if location:
                        location_match = re.search(r":(\d+):(\d+):\s*$", location)
                    if location_match:
                        clean_msg = f"line {location_match.group(1)}:{location_match.group(2)}: {clean_msg}"
                    if clean_msg and clean_msg not in errors:
                        errors.append(clean_msg)

        if not errors:
            tail = output.strip()[-1000:]
            errors.append(f"Unknown Compilation Error (Check logs): {tail}" if tail else "Unknown Compilation Error (Check logs)")

        return errors
