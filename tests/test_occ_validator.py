"""Optional integration checks against synthetic pythonocc shapes only."""
from __future__ import annotations

import pytest
from pathlib import Path
import tempfile

OCC = pytest.importorskip("OCC.Core.BRepPrimAPI")

from OCC.Core.BRep import BRep_Builder
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCC.Core.BRepTools import BRepTools_ReShape
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopoDS import TopoDS_Compound, TopoDS_Shape

from breprecover.validation.occ_validator import OCCTShapeValidator, validate_step_file
from breprecover.validation.isolated_step import IsolatedSTEPValidator


def _box(x: float = 1.0):
    return BRepPrimAPI_MakeBox(x, 2.0, 3.0).Shape()


def test_box_passes_four_checks_and_step_roundtrip(tmp_path):
    validator = OCCTShapeValidator()
    outcome = validator.inspect(_box())

    assert outcome.acceptance.kernel_valid is True
    assert outcome.acceptance.single_solid is True
    assert outcome.acceptance.boundary_edges_valid is True
    assert outcome.acceptance.step_roundtrip_valid is True
    assert outcome.acceptance.accepted is True
    assert outcome.details["step_transfer_status"]
    assert outcome.details["step_write_status"]
    assert outcome.details["step"]["transferred_roots"] > 0

    # The public file reader checks a persisted synthetic STEP independently.
    from OCC.Core.IFSelect import IFSelect_RetDone
    from OCC.Core.STEPControl import STEPControl_AsIs, STEPControl_Writer

    step = tmp_path / "box.step"
    writer = STEPControl_Writer()
    assert writer.Transfer(_box(), STEPControl_AsIs) == IFSelect_RetDone
    assert writer.Write(str(step)) == IFSelect_RetDone
    assert validate_step_file(step).roundtrip_valid is True
    isolated = IsolatedSTEPValidator(timeout_seconds=20).inspect_step(step)
    assert isolated.acceptance.accepted is True
    assert isolated.acceptance.kernel_valid is True
    assert isolated.acceptance.single_solid is True
    assert isolated.acceptance.boundary_edges_valid is True
    assert isolated.acceptance.step_roundtrip_valid is True


def test_compound_of_two_boxes_fails_single_solid_and_acceptance():
    builder = BRep_Builder()
    compound = TopoDS_Compound()
    builder.MakeCompound(compound)
    builder.Add(compound, _box())
    builder.Add(compound, BRepPrimAPI_MakeBox(4.0, 2.0, 3.0).Shape())

    outcome = OCCTShapeValidator().inspect(compound)
    assert outcome.details["solid_count"] == 2
    assert outcome.acceptance.kernel_valid is True
    assert outcome.acceptance.single_solid is False
    assert outcome.acceptance.accepted is False

    from OCC.Core.IFSelect import IFSelect_RetDone
    from OCC.Core.STEPControl import STEPControl_AsIs, STEPControl_Writer

    with tempfile.TemporaryDirectory() as directory:
        step = Path(directory) / "two-solids.step"
        writer = STEPControl_Writer()
        assert writer.Transfer(compound, STEPControl_AsIs) == IFSelect_RetDone
        assert writer.Write(str(step)) == IFSelect_RetDone
        isolated = IsolatedSTEPValidator(timeout_seconds=20).inspect_step(step)
    assert isolated.acceptance.kernel_valid is True
    assert isolated.acceptance.single_solid is False
    assert isolated.acceptance.accepted is False


def test_null_shape_and_missing_step_fail_closed(tmp_path):
    outcome = OCCTShapeValidator().inspect(TopoDS_Shape())
    assert outcome.acceptance.accepted is False
    assert outcome.details["shape"] == "null"

    missing = validate_step_file(tmp_path / "missing.step")
    assert missing.roundtrip_valid is False


def test_box_with_removed_face_reports_edge_metrics_and_is_rejected():
    box = _box()
    face_explorer = TopExp_Explorer(box, TopAbs_FACE)
    first_face = face_explorer.Current()
    reshaper = BRepTools_ReShape()
    reshaper.Remove(first_face)
    open_boundary_shape = reshaper.Apply(box)

    outcome = OCCTShapeValidator().inspect(open_boundary_shape)
    assert type(outcome.details["free_edge_count"]) is int
    assert type(outcome.details["bad_shell_count"]) is int
    assert outcome.details["boundary_edges_valid"] is (
        outcome.details["free_edge_count"] == 0
        and outcome.details["bad_shell_count"] == 0
    )
    assert outcome.acceptance.kernel_valid is False
    assert outcome.acceptance.accepted is False


def test_non_step_text_is_rejected(tmp_path):
    not_step = tmp_path / "not-step.step"
    not_step.write_text("this is not a STEP exchange file\n", encoding="ascii")
    assert validate_step_file(not_step).roundtrip_valid is False
