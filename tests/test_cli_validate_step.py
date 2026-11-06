"""CLI tests use only temporary, procedurally created STEP geometry."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

pytest.importorskip("OCC.Core.BRepPrimAPI")

from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.STEPControl import STEPControl_AsIs, STEPControl_Writer
from OCC.Core.BRep import BRep_Builder
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCC.Core.TopoDS import TopoDS_Compound

from breprecover.cli import main, validate_step
from breprecover.recovery.decision import AcceptanceReport
import breprecover.validation.isolated_step as isolated_step


SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"


def _write_step(shape, path: Path) -> Path:
    writer = STEPControl_Writer()
    assert writer.Transfer(shape, STEPControl_AsIs) == IFSelect_RetDone
    assert writer.Write(str(path)) == IFSelect_RetDone
    return path


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SOURCE_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-m", "breprecover.cli", *map(str, args)],
        env=env,
        stdin=subprocess.DEVNULL,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )


def test_validate_step_cli_accepts_box_and_writes_new_report(tmp_path):
    step = _write_step(BRepPrimAPI_MakeBox(1.0, 2.0, 3.0).Shape(), tmp_path / "box.step")
    report = tmp_path / "box-report.json"
    result = _run("validate-step", step, "--timeout", "20", "--report", report)

    assert result.returncode == 0, result.stderr
    stdout = json.loads(result.stdout)
    saved = json.loads(report.read_text(encoding="utf-8"))
    assert stdout == saved
    assert stdout["schema"] == "breprecover.step-validation-report.v1"
    assert stdout["accepted"] is True
    assert set(stdout["acceptance"]) == {
        "kernel_valid", "single_solid", "boundary_edges_valid", "step_roundtrip_valid"
    }


def test_validate_step_cli_rejects_multiple_solids_with_json_and_exit_one(tmp_path):
    builder = BRep_Builder()
    compound = TopoDS_Compound()
    builder.MakeCompound(compound)
    builder.Add(compound, BRepPrimAPI_MakeBox(1.0, 2.0, 3.0).Shape())
    builder.Add(compound, BRepPrimAPI_MakeBox(4.0, 2.0, 3.0).Shape())
    step = _write_step(compound, tmp_path / "two-solids.step")

    result = _run("validate-step", step, "--timeout", "20")

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["accepted"] is False
    assert payload["acceptance"]["single_solid"] is False


def test_validate_step_cli_rejects_bad_step_and_missing_path(tmp_path):
    malformed = tmp_path / "bad.step"
    malformed.write_text("not a STEP file\n", encoding="ascii")
    bad_result = _run("validate-step", malformed)
    assert bad_result.returncode == 1
    assert json.loads(bad_result.stdout)["accepted"] is False

    missing_result = _run("validate-step", tmp_path / "missing.step")
    assert missing_result.returncode == 2
    assert missing_result.stdout == ""


def test_report_never_overwrites_existing_path_or_input(tmp_path):
    step = _write_step(BRepPrimAPI_MakeBox(1.0, 2.0, 3.0).Shape(), tmp_path / "box.step")
    existing = tmp_path / "existing.json"
    existing.write_text("keep\n", encoding="ascii")

    existing_result = _run("validate-step", step, "--report", existing)
    assert existing_result.returncode == 2
    assert existing.read_text(encoding="ascii") == "keep\n"

    same_result = _run("validate-step", step, "--report", step)
    assert same_result.returncode == 2
    assert step.is_file()


def test_report_symlink_is_not_replaced(tmp_path):
    step = _write_step(BRepPrimAPI_MakeBox(1.0, 2.0, 3.0).Shape(), tmp_path / "box.step")
    target = tmp_path / "target.json"
    target.write_text("keep\n", encoding="ascii")
    link = tmp_path / "report-link.json"
    link.symlink_to(target)

    result = _run("validate-step", step, "--report", link)

    assert result.returncode == 2
    assert link.is_symlink()
    assert target.read_text(encoding="ascii") == "keep\n"


def test_malformed_timeout_is_usage_error():
    result = _run("validate-step", "input.step", "--timeout", "nan")
    assert result.returncode == 2
    assert "timeout_seconds" in result.stderr


@pytest.mark.parametrize(
    "details",
    [
        {"failure": "worker_timeout"},
        {"failure": "worker_start_failed"},
        {"failure": "worker_nonzero_exit"},
        {"failure": "worker_result_missing"},
        {"failure": "worker_result_invalid"},
        {"returncode": 0, "worker": {"exception": "RuntimeError"}},
        {"returncode": 0, "worker": {"shape_validation": {"exception": "ImportError"}}},
        {
            "returncode": 0,
            "worker": {"shape_validation": {"step": {"exception": "OSError"}}},
        },
    ],
)
def test_worker_operational_failures_are_distinguished_from_geometry_rejection(
    tmp_path, monkeypatch, details
):
    step = tmp_path / "existing.step"
    step.write_text("placeholder", encoding="ascii")

    class FakeValidator:
        def __init__(self, **kwargs):
            pass

        def inspect_step(self, path):
            return isolated_step.IsolatedSTEPOutcome(
                AcceptanceReport(False, False, False, False), details
            )

    monkeypatch.setattr(isolated_step, "IsolatedSTEPValidator", FakeValidator)
    report = validate_step(step)
    assert report["accepted"] is False
    assert report["runtime_error"] is True


def test_completed_geometric_rejection_is_not_runtime_error(tmp_path, monkeypatch):
    step = tmp_path / "existing.step"
    step.write_text("placeholder", encoding="ascii")

    class FakeValidator:
        def __init__(self, **kwargs):
            pass

        def inspect_step(self, path):
            return isolated_step.IsolatedSTEPOutcome(
                AcceptanceReport(True, False, True, True),
                {"returncode": 0, "worker": {"input_step_transferred_roots": 1}},
            )

    monkeypatch.setattr(isolated_step, "IsolatedSTEPValidator", FakeValidator)
    report = validate_step(step)
    assert report["accepted"] is False
    assert report["runtime_error"] is False


@pytest.mark.parametrize(
    "details,expected_code,expected_runtime_error",
    [
        ({"failure": "worker_timeout"}, 2, True),
        ({"failure": "worker_result_invalid"}, 2, True),
        ({"returncode": 0, "worker": {"input_step_read_status": "bad"}}, 1, False),
    ],
)
def test_cli_exit_codes_and_stdout_json_for_worker_outcomes(
    tmp_path, monkeypatch, capsys, details, expected_code, expected_runtime_error
):
    step = tmp_path / "existing.step"
    step.write_text("placeholder", encoding="ascii")

    class FakeValidator:
        def __init__(self, **kwargs):
            pass

        def inspect_step(self, path):
            if details.get("failure") == "worker_timeout":
                checks = AcceptanceReport(False, False, False, False)
            elif details.get("failure") == "worker_result_invalid":
                checks = AcceptanceReport(False, False, False, False)
            else:
                checks = AcceptanceReport(False, False, False, False)
            return isolated_step.IsolatedSTEPOutcome(checks, details)

    monkeypatch.setattr(isolated_step, "IsolatedSTEPValidator", FakeValidator)
    code = main(["validate-step", str(step)])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert code == expected_code
    assert payload["runtime_error"] is expected_runtime_error
    assert captured.err == ""
