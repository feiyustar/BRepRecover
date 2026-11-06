"""Optional OCCT acceptance checks implemented with public pythonocc APIs.

This module validates a supplied shape; it performs no healing or construction.
Its four acceptance checks follow the semantics documented in
docs/backend_contract.md. Provenance and third-party licensing still require
project-level review.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tempfile
from typing import Any

from breprecover.recovery.decision import AcceptanceReport


@dataclass(frozen=True)
class StepFileOutcome:
    roundtrip_valid: bool
    details: dict[str, Any]


@dataclass(frozen=True)
class OCCValidationOutcome:
    acceptance: AcceptanceReport
    details: dict[str, Any]


def _occ_api():
    try:
        from OCC.Core.BRepCheck import BRepCheck_Analyzer
        from OCC.Core.IFSelect import IFSelect_RetDone
        from OCC.Core.ShapeAnalysis import ShapeAnalysis_FreeBounds, ShapeAnalysis_Shell
        from OCC.Core.STEPControl import STEPControl_AsIs, STEPControl_Reader, STEPControl_Writer
        from OCC.Core.TopoDS import topods_Shell
        from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_SHELL, TopAbs_SOLID
        from OCC.Core.TopExp import TopExp_Explorer
    except ImportError as exc:
        raise RuntimeError("optional pythonocc-core/OCCT runtime is unavailable") from exc
    return {
        "BRepCheck_Analyzer": BRepCheck_Analyzer,
        "IFSelect_RetDone": IFSelect_RetDone,
        "ShapeAnalysis_FreeBounds": ShapeAnalysis_FreeBounds,
        "ShapeAnalysis_Shell": ShapeAnalysis_Shell,
        "STEPControl_AsIs": STEPControl_AsIs,
        "STEPControl_Reader": STEPControl_Reader,
        "STEPControl_Writer": STEPControl_Writer,
        "topods_Shell": topods_Shell,
        "TopAbs_EDGE": TopAbs_EDGE,
        "TopAbs_SHELL": TopAbs_SHELL,
        "TopAbs_SOLID": TopAbs_SOLID,
        "TopExp_Explorer": TopExp_Explorer,
    }


def _count(shape: Any, kind: Any, explorer_type: Any) -> int:
    explorer = explorer_type(shape, kind)
    count = 0
    while explorer.More():
        count += 1
        explorer.Next()
    return count


def _shape_metrics(shape: Any, api: dict[str, Any]) -> dict[str, Any]:
    valid = bool(api["BRepCheck_Analyzer"](shape).IsValid())
    solids = _count(shape, api["TopAbs_SOLID"], api["TopExp_Explorer"])
    open_wires = api["ShapeAnalysis_FreeBounds"](shape).GetOpenWires()
    free_edges = _count(open_wires, api["TopAbs_EDGE"], api["TopExp_Explorer"])
    shell_explorer = api["TopExp_Explorer"](shape, api["TopAbs_SHELL"])
    bad_shells = 0
    while shell_explorer.More():
        analysis = api["ShapeAnalysis_Shell"]()
        analysis.LoadShells(api["topods_Shell"](shell_explorer.Current()))
        bad_shells += int(analysis.HasBadEdges())
        shell_explorer.Next()
    return {
        "kernel_valid": valid,
        "solid_count": int(solids),
        "free_edge_count": int(free_edges),
        "bad_shell_count": int(bad_shells),
        "boundary_edges_valid": bool(free_edges == 0 and bad_shells == 0),
    }


def validate_step_file(path: str | Path) -> StepFileOutcome:
    """Read a STEP file and require status, transfer roots, and strict topology."""
    file_path = Path(path)
    details: dict[str, Any] = {"path": str(file_path), "exists": False}
    try:
        details["exists"] = file_path.is_file()
        if not details["exists"] or file_path.stat().st_size == 0:
            details["read_status"] = "missing_or_empty"
            return StepFileOutcome(False, details)
        api = _occ_api()
        reader = api["STEPControl_Reader"]()
        read_status = reader.ReadFile(str(file_path))
        details["read_status"] = str(read_status)
        if read_status != api["IFSelect_RetDone"]:
            return StepFileOutcome(False, details)
        transferred_roots = int(reader.TransferRoots())
        details["transferred_roots"] = transferred_roots
        if transferred_roots <= 0:
            return StepFileOutcome(False, details)
        loaded = reader.OneShape()
        if loaded is None or loaded.IsNull():
            details["one_shape_null"] = True
            return StepFileOutcome(False, details)
        metrics = _shape_metrics(loaded, api)
        details["roundtrip_shape"] = metrics
        accepted = bool(
            metrics["kernel_valid"]
            and metrics["solid_count"] == 1
            and metrics["boundary_edges_valid"]
        )
        return StepFileOutcome(accepted, details)
    except Exception as exc:
        details["exception"] = f"{type(exc).__name__}: {exc}"
        return StepFileOutcome(False, details)


class OCCTShapeValidator:
    """Independent kernel/topology/STEP validation; no CAD construction."""

    def inspect(self, shape: Any) -> OCCValidationOutcome:
        if shape is None:
            return OCCValidationOutcome(
                AcceptanceReport(False, False, False, False),
                {"shape": "none", "step_status": "skipped"},
            )
        try:
            api = _occ_api()
            if shape.IsNull():
                return OCCValidationOutcome(
                    AcceptanceReport(False, False, False, False),
                    {"shape": "null", "step_status": "skipped"},
                )
            metrics = _shape_metrics(shape, api)
            with tempfile.TemporaryDirectory(prefix="breprecover-occt-check-") as directory:
                step_path = Path(directory) / "roundtrip.step"
                writer = api["STEPControl_Writer"]()
                transfer_status = writer.Transfer(shape, api["STEPControl_AsIs"])
                write_status = None
                if transfer_status == api["IFSelect_RetDone"]:
                    write_status = writer.Write(str(step_path))
                write_ok = bool(
                    transfer_status == api["IFSelect_RetDone"]
                    and write_status == api["IFSelect_RetDone"]
                    and step_path.is_file()
                    and step_path.stat().st_size > 0
                )
                step_outcome = validate_step_file(step_path) if write_ok else StepFileOutcome(
                    False,
                    {
                        "transfer_status": str(transfer_status),
                        "write_status": None if write_status is None else str(write_status),
                        "exists": step_path.is_file(),
                    },
                )
            acceptance = AcceptanceReport(
                bool(metrics["kernel_valid"]),
                bool(metrics["solid_count"] == 1),
                bool(metrics["boundary_edges_valid"]),
                bool(step_outcome.roundtrip_valid),
            )
            return OCCValidationOutcome(
                acceptance,
                {
                    **metrics,
                    "step_transfer_status": str(transfer_status),
                    "step_write_status": None if write_status is None else str(write_status),
                    "step": step_outcome.details,
                },
            )
        except Exception as exc:
            return OCCValidationOutcome(
                AcceptanceReport(False, False, False, False),
                {"exception": f"{type(exc).__name__}: {exc}"},
            )

    def validate(self, candidate: Any) -> AcceptanceReport:
        """Implement the shared IndependentValidator structural protocol."""
        return self.inspect(candidate).acceptance
