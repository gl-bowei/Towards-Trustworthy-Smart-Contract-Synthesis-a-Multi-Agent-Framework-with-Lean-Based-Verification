import json
import os
import shlex
import shutil
import subprocess
from typing import Dict

IGNORED_SLITHER_DETECTORS = {
    "solc-version",
    "naming-convention",
    "assembly",
    "pragma",
    "unused-return",
    "dead-code",
    "redundant-statement",
    "costly-loop",
}

def run_slither_check(solidity_path: str) -> Dict:
    file_dir = os.path.dirname(solidity_path)
    file_name = os.path.basename(solidity_path)
    output_name = "slither_core_metrics.json"
    output_path = os.path.join(file_dir, output_name)

    configured_command = os.getenv("SLITHER_COMMAND", "slither")
    command = shlex.split(configured_command)
    if not command:
        return {"slither_pass": None, "infra_broken": True, "error": "SLITHER_COMMAND is empty"}
    executable = shutil.which(command[0])
    if executable is None:
        return {
            "slither_pass": None,
            "infra_broken": True,
            "error": f"slither command not found: {command[0]}",
        }
    command[0] = executable

    try:
        proc = subprocess.run(
            [*command, file_name, "--json", output_name],
            cwd=file_dir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=60,
        )
    except FileNotFoundError:
        return {"slither_pass": None, "infra_broken": True, "error": "slither command not found"}
    except subprocess.TimeoutExpired:
        return {"slither_pass": None, "infra_broken": True, "error": "slither timed out"}
    except Exception as e:
        return {"slither_pass": None, "infra_broken": True, "error": str(e)}

    if not os.path.exists(output_path):
        return {
            "slither_pass": None,
            "infra_broken": True,
            "error": f"slither did not produce JSON output (exit={proc.returncode}): {proc.stderr[-1000:]}",
        }

    try:
        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return {"slither_pass": None, "infra_broken": True, "error": f"failed to parse slither JSON: {e}"}
    finally:
        try:
            os.remove(output_path)
        except OSError:
            pass

    counts = {"high": 0, "medium": 0, "low": 0}
    for det in data.get("results", {}).get("detectors", []):
        if det.get("check") in IGNORED_SLITHER_DETECTORS:
            continue
        impact = str(det.get("impact", "")).lower()
        if impact in counts:
            counts[impact] += 1

    return {
        "slither_pass": counts["high"] == 0,
        "high": counts["high"],
        "medium": counts["medium"],
        "low": counts["low"],
        "infra_broken": False,
        "error": "",
    }
