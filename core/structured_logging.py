import datetime
import json
import os
import re
import time
from typing import Any, Dict, Optional


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
ADDRESS_RE = re.compile(r"0x[a-fA-F0-9]{40}")
PRIVATE_KEY_ARG_RE = re.compile(r"(--private-key\s+)(0x)?[a-fA-F0-9]{64}")
OPENAI_KEY_RE = re.compile(r"sk-[A-Za-z0-9_-]{12,}")


def utc_timestamp() -> str:
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def clean_ansi(text: str) -> str:
    return ANSI_RE.sub("", text or "")


def redact_secrets(text: str) -> str:
    text = text or ""
    text = PRIVATE_KEY_ARG_RE.sub(r"\1[REDACTED_PRIVATE_KEY]", text)
    text = OPENAI_KEY_RE.sub("[REDACTED_API_KEY]", text)
    return text


def text_tail(text: str, limit: int = 4000) -> str:
    text = redact_secrets(clean_ansi(text or ""))
    if len(text) <= limit:
        return text
    return text[-limit:]


def compact_error(text: str, limit: int = 1200) -> str:
    text = redact_secrets(clean_ansi(text or "")).strip()
    if not text:
        return ""
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    preferred = []
    for line in lines:
        lowered = line.lower()
        if (
            "error" in lowered
            or "fail" in lowered
            or "panic" in lowered
            or "revert" in lowered
            or "compiler run failed" in lowered
        ):
            preferred.append(line)
    summary = "\n".join(preferred[:12]) if preferred else "\n".join(lines[:12])
    if len(summary) > limit:
        summary = summary[:limit] + "\n...[truncated]"
    return summary


def parse_foundry_output(stdout: str, stderr: str, returncode: int) -> Dict[str, Any]:
    stdout = redact_secrets(clean_ansi(stdout or ""))
    stderr = redact_secrets(clean_ansi(stderr or ""))
    combined = f"{stdout}\n{stderr}"

    tags = {}
    for line in combined.splitlines():
        if "TAG:" not in line and ":" not in line:
            continue
        addresses = ADDRESS_RE.findall(line)
        if not addresses:
            continue
        tag_match = re.search(r"TAG:\s*([A-Za-z0-9_:-]+)", line)
        if tag_match:
            tags[tag_match.group(1).strip(":")] = addresses[-1]
            continue
        plain_match = re.search(r"\b([A-Z][A-Z0-9_]{2,})\b\s*:?\s*0x[a-fA-F0-9]{40}", line)
        if plain_match:
            tags[plain_match.group(1)] = addresses[-1]

    pass_tests = [
        m.groups()
        for m in re.finditer(
            r"\[PASS\]\s+([^\n(]+(?:\([^\n]*\))?)\s+\(runs:\s*(\d+)(?:,\s*calls:\s*(\d+))?",
            combined,
        )
    ]
    fail_tests = [m.group(1).strip() for m in re.finditer(r"\[FAIL[^\]]*\]\s+([^\n]+)", combined)]
    suite_match = re.search(
        r"Suite result:\s*(\w+)\.\s*(\d+)\s+passed;\s*(\d+)\s+failed;\s*(\d+)\s+skipped",
        combined,
    )
    ran_match = re.search(
        r"Ran\s+(\d+)\s+test suite.*?:\s*(\d+)\s+tests passed,\s*(\d+)\s+failed,\s*(\d+)\s+skipped",
        combined,
    )
    runs = [int(x) for x in re.findall(r"runs:\s*(\d+)", combined)]
    test_call_counts = [int(calls) for _, _, calls in pass_tests if calls]
    if test_call_counts:
        fuzz_calls = sum(test_call_counts)
    else:
        handler_rows = re.findall(
            r"\|\s*Handler\s*\|\s*[^|]+\|\s*(\d+)\s*\|\s*(\d+)\s*\|",
            combined,
        )
        fuzz_calls = sum(int(calls) for calls, _ in handler_rows)
    failing_section = combined.split("Failing tests:", 1)[1] if "Failing tests:" in combined else ""
    broken_invariants = set()
    for failed_name in fail_tests:
        match = re.search(r"\b(invariant_[A-Za-z0-9_]+)\s*(?:\(|$)", failed_name)
        if match:
            broken_invariants.add(match.group(1))
    if returncode != 0 and failing_section:
        broken_invariants.update(re.findall(r"\b(invariant_[A-Za-z0-9_]+)\(\)\s+\(runs:", failing_section))

    revert_reason = ""
    for pattern in (
        r"Error:\s*script failed:\s*([^\n]+)",
        r"Revert reason:\s*([^\n]+)",
        r"\[FAIL\.\s*Reason:\s*([^\]]+)\]",
    ):
        match = re.search(pattern, combined)
        if match:
            revert_reason = match.group(1).strip()
            break

    trace_excerpt = ""
    if "Traces:" in combined:
        trace_excerpt = "Traces:" + combined.split("Traces:", 1)[1][:3000]

    compiler_failed = "Compiler run failed" in combined or "Error (" in combined
    script_success = "Script ran successfully" in combined or "ONCHAIN EXECUTION COMPLETE" in combined
    broadcast_success = "ONCHAIN EXECUTION COMPLETE" in combined
    simulated_execution_failed = "Error: Simulated execution failed" in combined
    insufficient_funds = "Insufficient funds for gas * price + value" in combined
    foundry_panic = "The application panicked" in combined

    if returncode == 0 and broadcast_success:
        execution_status = "broadcast_confirmed"
    elif returncode == 0 and script_success:
        execution_status = "simulation_success"
    elif compiler_failed:
        execution_status = "compiler_failed"
    elif foundry_panic:
        execution_status = "foundry_panic"
    elif insufficient_funds:
        execution_status = "broadcast_failed_insufficient_funds"
    elif simulated_execution_failed:
        execution_status = "simulation_failed_or_reverted"
    elif script_success and not broadcast_success:
        execution_status = "simulation_success_broadcast_failed"
    elif returncode != 0:
        execution_status = "command_failed"
    else:
        execution_status = "unknown"

    summary = {
        "returncode": returncode,
        "success": returncode == 0,
        "compiler_success": "Compiler run successful" in combined,
        "compiler_failed": compiler_failed,
        "script_success": script_success,
        "broadcast_success": broadcast_success,
        "simulated_execution_failed": simulated_execution_failed,
        "insufficient_funds": insufficient_funds,
        "execution_status": execution_status,
        "foundry_panic": foundry_panic,
        "panic_message": "",
        "deployed_tags": tags,
        "passed_tests": [name for name, _, _ in pass_tests],
        "failed_tests": fail_tests,
        "passed_test_count": len(pass_tests),
        "failed_test_count": len(fail_tests),
        "fuzz_runs": sum(runs),
        "fuzz_calls": fuzz_calls,
        "broken_invariants": sorted(broken_invariants),
        "revert_reason": revert_reason,
        "error_summary": compact_error(stderr if returncode else combined),
        "trace_excerpt": trace_excerpt,
    }

    panic_match = re.search(r"Message:\s*(.+)", combined)
    if panic_match:
        summary["panic_message"] = panic_match.group(1).strip()

    if suite_match:
        summary["suite_result"] = {
            "status": suite_match.group(1),
            "passed": int(suite_match.group(2)),
            "failed": int(suite_match.group(3)),
            "skipped": int(suite_match.group(4)),
        }
        summary["passed_test_count"] = summary["suite_result"]["passed"]
        summary["failed_test_count"] = summary["suite_result"]["failed"]
    if ran_match:
        summary["run_result"] = {
            "suites": int(ran_match.group(1)),
            "passed": int(ran_match.group(2)),
            "failed": int(ran_match.group(3)),
            "skipped": int(ran_match.group(4)),
        }
        if not suite_match:
            summary["passed_test_count"] = summary["run_result"]["passed"]
            summary["failed_test_count"] = summary["run_result"]["failed"]

    if returncode == 0 or int(summary.get("failed_test_count") or 0) == 0:
        summary["broken_invariants"] = []

    return summary


class JsonlEventLogger:
    def __init__(self, path: str, metadata: Optional[Dict[str, Any]] = None):
        self.path = os.path.abspath(path)
        self.metadata = metadata or {}
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

    def emit(self, event_type: str, payload: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Dict[str, Any]:
        event = {
            "ts": utc_timestamp(),
            "event": event_type,
            "metadata": self.metadata,
            "payload": payload or {},
        }
        if kwargs:
            event["payload"].update(kwargs)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
        return event


class CommandRecorder:
    def __init__(self, log_dir: str, event_logger: Optional[JsonlEventLogger] = None):
        self.log_dir = os.path.abspath(log_dir)
        self.event_logger = event_logger
        self.counter = 0
        os.makedirs(self.log_dir, exist_ok=True)

    def attach_event_logger(self, event_logger: Optional[JsonlEventLogger]):
        self.event_logger = event_logger

    def record(
        self,
        *,
        cmd: str,
        returncode: int,
        stdout: str,
        stderr: str,
        cwd: str,
        started_at: float,
        ended_at: float,
        label: Optional[str] = None,
        kind: str = "command",
    ) -> Dict[str, Any]:
        self.counter += 1
        slug_source = label or kind or "command"
        slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", slug_source).strip("_")[:80] or "command"
        prefix = f"{self.counter:04d}_{slug}"
        stdout_path = os.path.join(self.log_dir, f"{prefix}.stdout.log")
        stderr_path = os.path.join(self.log_dir, f"{prefix}.stderr.log")
        meta_path = os.path.join(self.log_dir, f"{prefix}.json")

        os.makedirs(self.log_dir, exist_ok=True)
        with open(stdout_path, "w", encoding="utf-8") as f:
            f.write(stdout or "")
        with open(stderr_path, "w", encoding="utf-8") as f:
            f.write(stderr or "")

        foundry = parse_foundry_output(stdout, stderr, returncode) if "forge " in cmd or cmd.startswith("forge") else {}
        record = {
            "label": label or slug,
            "kind": kind,
            "cmd": redact_secrets(cmd),
            "cwd": os.path.abspath(cwd),
            "returncode": returncode,
            "success": returncode == 0,
            "duration_sec": round(ended_at - started_at, 3),
            "started_at": datetime.datetime.fromtimestamp(started_at).isoformat(timespec="seconds"),
            "ended_at": datetime.datetime.fromtimestamp(ended_at).isoformat(timespec="seconds"),
            "stdout_path": stdout_path,
            "stderr_path": stderr_path,
            "stdout_tail": text_tail(stdout, 3000),
            "stderr_tail": text_tail(stderr, 3000),
            "foundry": foundry,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)
        record["meta_path"] = meta_path

        if self.event_logger:
            event_payload = record.copy()
            event_payload.pop("stdout_tail", None)
            event_payload.pop("stderr_tail", None)
            self.event_logger.emit("command", event_payload)
        return record
