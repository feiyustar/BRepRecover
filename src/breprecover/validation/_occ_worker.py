"""Internal JSON-only OCCT worker; never accepts pickle or shape serialization."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

_REQUEST_SCHEMA = "breprecover.occt-validation-request.v1"
_RESULT_SCHEMA = "breprecover.occt-validation-result.v1"
_PREDICATES = (
    "kernel_valid",
    "single_solid",
    "boundary_edges_valid",
    "step_roundtrip_valid",
)


def _load_request(path: Path) -> tuple[Path, Path]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != {"schema", "input_step", "result_json"}:
        raise ValueError("invalid worker request fields")
    if payload["schema"] != _REQUEST_SCHEMA:
        raise ValueError("worker request schema mismatch")
    if not isinstance(payload["input_step"], str) or not isinstance(payload["result_json"], str):
        raise ValueError("worker paths must be strings")
    input_step = Path(payload["input_step"])
    result_json = Path(payload["result_json"])
    if not input_step.is_absolute() or not result_json.is_absolute():
        raise ValueError("worker paths must be absolute")
    if result_json.parent.resolve() != path.parent.resolve():
        raise ValueError("result path must remain inside worker workspace")
    return input_step, result_json


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    encoded = json.dumps(payload, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    with temporary.open("wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _failure(details: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": _RESULT_SCHEMA,
        "acceptance": {name: False for name in _PREDICATES},
        "details": details,
    }


def _validate(input_step: Path) -> dict[str, Any]:
    from breprecover.validation.occ_validator import OCCTShapeValidator, _occ_api

    if not input_step.is_file() or input_step.stat().st_size == 0:
        return _failure({"input_step": str(input_step), "failure": "missing_or_empty"})
    api = _occ_api()
    reader = api["STEPControl_Reader"]()
    read_status = reader.ReadFile(str(input_step))
    details: dict[str, Any] = {"input_step_read_status": str(read_status)}
    if read_status != api["IFSelect_RetDone"]:
        return _failure(details)
    transferred_roots = int(reader.TransferRoots())
    details["input_step_transferred_roots"] = transferred_roots
    if transferred_roots <= 0:
        return _failure(details)
    shape = reader.OneShape()
    if shape is None or shape.IsNull():
        details["input_shape_null"] = True
        return _failure(details)

    outcome = OCCTShapeValidator().inspect(shape)
    acceptance = {
        "kernel_valid": outcome.acceptance.kernel_valid,
        "single_solid": outcome.acceptance.single_solid,
        "boundary_edges_valid": outcome.acceptance.boundary_edges_valid,
        "step_roundtrip_valid": outcome.acceptance.step_roundtrip_valid,
    }
    details["shape_validation"] = outcome.details
    return {"schema": _RESULT_SCHEMA, "acceptance": acceptance, "details": details}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--request", required=True)
    args = parser.parse_args(argv)
    request_path = Path(args.request)
    try:
        input_step, result_path = _load_request(request_path)
        try:
            result = _validate(input_step)
        except Exception as exc:
            result = _failure({"exception": f"{type(exc).__name__}: {exc}"})
        _atomic_json(result_path, result)
        return 0
    except Exception:
        return 2


if __name__ == "__main__":
    sys.exit(main())
