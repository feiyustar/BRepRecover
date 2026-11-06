"""Process-isolated validation of an existing STEP path (not a shape object)."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
from typing import Any

from breprecover.recovery.decision import AcceptanceReport

_REQUEST_SCHEMA = "breprecover.occt-validation-request.v1"
_RESULT_SCHEMA = "breprecover.occt-validation-result.v1"
_MAX_RESULT_BYTES = 64 * 1024
_PREDICATES = (
    "kernel_valid",
    "single_solid",
    "boundary_edges_valid",
    "step_roundtrip_valid",
)


def _rejected(reason: str, **details: Any) -> "IsolatedSTEPOutcome":
    return IsolatedSTEPOutcome(
        AcceptanceReport(False, False, False, False),
        {"failure": reason, **details},
    )


class IsolatedSTEPOutcome:
    """Four worker-reported predicates and supervisor diagnostics."""

    def __init__(self, acceptance: AcceptanceReport, details: dict[str, Any]):
        self.acceptance = acceptance
        self.details = details


def _kill_and_wait(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        process.wait()
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=0.25)
    except subprocess.TimeoutExpired:
        pass
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            if process.poll() is None:
                process.kill()
    except ProcessLookupError:
        pass
    process.wait()


def _read_worker_result(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        encoded = stream.read(_MAX_RESULT_BYTES + 1)
    if not encoded or len(encoded) > _MAX_RESULT_BYTES:
        raise ValueError(f"worker result size outside 1..{_MAX_RESULT_BYTES} bytes")
    payload = json.loads(encoded.decode("utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != _RESULT_SCHEMA:
        raise ValueError("worker result schema mismatch")
    if set(payload) != {"schema", "acceptance", "details"}:
        raise ValueError("worker result has missing or unexpected fields")
    checks = payload["acceptance"]
    if not isinstance(checks, dict) or set(checks) != set(_PREDICATES):
        raise ValueError("worker acceptance predicates are incomplete")
    if any(type(checks[name]) is not bool for name in _PREDICATES):
        raise ValueError("worker predicates must be built-in bool values")
    if not isinstance(payload["details"], dict):
        raise ValueError("worker details must be an object")
    return payload


def _run_worker(
    step_path: str | Path,
    *,
    timeout_seconds: float,
    python_executable: str,
    worker_module: str,
) -> IsolatedSTEPOutcome:
    """Private seam allows tests to use deterministic fake Python workers."""
    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)):
        raise ValueError("timeout_seconds must be a positive finite number")
    timeout_seconds = float(timeout_seconds)
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be a positive finite number")
    if not isinstance(python_executable, str) or not python_executable:
        raise ValueError("python_executable must be a nonempty path")

    input_path = os.path.abspath(os.path.expanduser(os.fspath(step_path)))
    package_src = str(Path(__file__).resolve().parents[2])
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = package_src if not existing else package_src + os.pathsep + existing

    with tempfile.TemporaryDirectory(prefix="breprecover-step-worker-") as directory:
        workspace = Path(directory)
        worker_tmp = workspace / "worker-tmp"
        worker_tmp.mkdir()
        env["TMPDIR"] = str(worker_tmp)
        env["TEMP"] = str(worker_tmp)
        env["TMP"] = str(worker_tmp)
        request_path = workspace / "request.json"
        result_path = workspace / "result.json"
        request_path.write_text(
            json.dumps(
                {
                    "schema": _REQUEST_SCHEMA,
                    "input_step": input_path,
                    "result_json": str(result_path),
                },
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
        command = [python_executable, "-m", worker_module, "--request", str(request_path)]
        try:
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=env,
                start_new_session=(os.name == "posix"),
            )
        except Exception as exc:
            return _rejected("worker_start_failed", exception=f"{type(exc).__name__}: {exc}")

        try:
            returncode = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            _kill_and_wait(process)
            return _rejected("worker_timeout", timeout_seconds=timeout_seconds)
        except BaseException:
            _kill_and_wait(process)
            raise
        if returncode != 0:
            return _rejected("worker_nonzero_exit", returncode=returncode)
        if not result_path.is_file():
            return _rejected("worker_result_missing", returncode=returncode)
        try:
            payload = _read_worker_result(result_path)
            acceptance = AcceptanceReport(**payload["acceptance"])
        except Exception as exc:
            return _rejected("worker_result_invalid", exception=f"{type(exc).__name__}: {exc}")
        return IsolatedSTEPOutcome(
            acceptance,
            {"returncode": returncode, "worker": payload["details"]},
        )


class IsolatedSTEPValidator:
    """Validate a STEP file in a worker; intentionally not a shape validator.

    The parent accepts only a path, never serializes an arbitrary OCC shape,
    and recomputes overall acceptance from four strictly typed predicates.
    """

    def __init__(
        self,
        *,
        python_executable: str | Path | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)):
            raise ValueError("timeout_seconds must be a positive finite number")
        timeout_seconds = float(timeout_seconds)
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be a positive finite number")
        self.python_executable = str(python_executable or sys.executable)
        self.timeout_seconds = timeout_seconds

    def inspect_step(self, path: str | Path) -> IsolatedSTEPOutcome:
        return _run_worker(
            path,
            timeout_seconds=self.timeout_seconds,
            python_executable=self.python_executable,
            worker_module="breprecover.validation._occ_worker",
        )

    def validate_step(self, path: str | Path) -> AcceptanceReport:
        return self.inspect_step(path).acceptance
