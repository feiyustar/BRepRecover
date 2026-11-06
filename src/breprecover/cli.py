"""Command-line checks for configuration and existing STEP files."""
from __future__ import annotations

import argparse
import json
import math
import numbers
import os
from pathlib import Path
import tempfile
from typing import Any

REQUIRED_ARTIFACTS = (
    "generator_checkpoint",
    "candidate_router_checkpoint",
    "shape_selector_checkpoint",
    "feature_runtime_checkpoint",
    "feature_runtime_config",
    "shape_policy_metadata",
)


def validate_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    artifacts = config.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("configuration must contain an artifacts object")
    missing = [key for key in REQUIRED_ARTIFACTS if not artifacts.get(key)]
    if missing:
        raise ValueError(f"missing required artifact paths: {', '.join(missing)}")
    resolved: dict[str, str] = {}
    for key in REQUIRED_ARTIFACTS:
        artifact = Path(artifacts[key]).expanduser()
        if not artifact.is_absolute():
            artifact = config_path.parent / artifact
        artifact = artifact.resolve()
        if not artifact.is_file():
            raise FileNotFoundError(f"required artifact {key} not found: {artifact}")
        resolved[key] = str(artifact)
    metadata = json.loads(Path(resolved["shape_policy_metadata"]).read_text(encoding="utf-8"))
    frozen_threshold = metadata.get("selected_dev_policy", {}).get("threshold")
    policy = config.get("policy")
    if not isinstance(policy, dict):
        raise ValueError("configuration must contain a policy object")
    configured_threshold = policy.get("shape_threshold")
    for name, value in (
        ("policy.shape_threshold", configured_threshold),
        ("selected_dev_policy.threshold", frozen_threshold),
    ):
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise ValueError(f"{name} must be a numeric threshold, not a bool or string")
    frozen_threshold = float(frozen_threshold)
    configured_threshold = float(configured_threshold)
    if not math.isfinite(frozen_threshold) or not math.isfinite(configured_threshold):
        raise ValueError("shape thresholds must be finite")
    if not 0.0 <= configured_threshold <= 1.0:
        raise ValueError("policy.shape_threshold must be in [0, 1]")
    if abs(configured_threshold - frozen_threshold) > 1e-9:
        raise ValueError("configured shape threshold does not match frozen metadata")
    return {
        "valid": True,
        "config": str(config_path.resolve()),
        "artifacts": resolved,
        "shape_threshold": frozen_threshold,
        "threshold_source": "shape_policy_metadata.selected_dev_policy.threshold",
    }


def validate_step(
    path: str | Path,
    *,
    timeout_seconds: float = 30.0,
    python_executable: str | None = None,
) -> dict[str, Any]:
    """Validate an existing STEP file in an isolated OCCT worker."""
    from breprecover.validation.isolated_step import IsolatedSTEPValidator

    validator = IsolatedSTEPValidator(
        python_executable=python_executable,
        timeout_seconds=timeout_seconds,
    )
    step_path = Path(path).expanduser()
    if not step_path.is_file():
        raise FileNotFoundError(f"STEP input is not a regular file: {step_path}")
    outcome = validator.inspect_step(step_path)
    acceptance = outcome.acceptance
    checks = {
        "kernel_valid": acceptance.kernel_valid,
        "single_solid": acceptance.single_solid,
        "boundary_edges_valid": acceptance.boundary_edges_valid,
        "step_roundtrip_valid": acceptance.step_roundtrip_valid,
    }
    details = outcome.details
    worker_details = details.get("worker") if isinstance(details, dict) else None
    shape_details = worker_details.get("shape_validation") if isinstance(worker_details, dict) else None
    step_details = shape_details.get("step") if isinstance(shape_details, dict) else None
    operational_failures = {
        "worker_start_failed",
        "worker_timeout",
        "worker_nonzero_exit",
        "worker_result_missing",
        "worker_result_invalid",
    }
    runtime_error = (
        isinstance(details, dict) and details.get("failure") in operational_failures
    ) or (isinstance(worker_details, dict) and "exception" in worker_details) or (
        isinstance(shape_details, dict) and "exception" in shape_details
    ) or (
        isinstance(step_details, dict) and "exception" in step_details
    )
    return {
        "schema": "breprecover.step-validation-report.v1",
        "input_step": str(step_path.resolve()),
        "accepted": all(value is True for value in checks.values()),
        "runtime_error": runtime_error,
        "acceptance": checks,
        "details": details,
    }


def write_report_no_replace(path: str | Path, payload: dict[str, Any], input_path: str | Path) -> None:
    """Atomically create a JSON report without overwriting any existing path."""
    report_path = Path(path).expanduser().absolute()
    source_path = Path(input_path).expanduser().resolve(strict=True)
    if report_path.resolve(strict=False) == source_path:
        raise ValueError("report path must not refer to the input STEP file")
    if os.path.lexists(report_path):
        raise FileExistsError(f"report path already exists: {report_path}")
    parent = report_path.parent
    if not parent.is_dir():
        raise FileNotFoundError(f"report directory does not exist: {parent}")
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{report_path.name}.", suffix=".tmp",
            dir=parent, delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        # A hard-link install is atomic and fails if another process created
        # the destination after the initial lexists check.
        os.link(temporary_path, report_path, follow_symlinks=False)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="breprecover")
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("validate-config", help="check required external artifact paths")
    check.add_argument("config", type=Path)
    step = commands.add_parser("validate-step", help="validate an existing STEP file; does not repair it")
    step.add_argument("path", type=Path)
    step.add_argument("--timeout", type=float, default=30.0, dest="timeout_seconds")
    step.add_argument("--python-executable", type=str, default=None)
    step.add_argument("--report", type=Path, default=None, help="create a new JSON report without overwriting")
    args = parser.parse_args(argv)
    try:
        if args.command == "validate-config":
            result = validate_config(args.config)
        else:
            result = validate_step(
                args.path,
                timeout_seconds=args.timeout_seconds,
                python_executable=args.python_executable,
            )
            if args.report is not None:
                write_report_no_replace(args.report, result, args.path)
    except Exception as exc:
        parser.exit(2, f"breprecover: {type(exc).__name__}: {exc}\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.command == "validate-step":
        if result["runtime_error"]:
            return 2
        if not result["accepted"]:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
