from __future__ import annotations

import ast
import json
import os
from pathlib import Path
from typing import Any, Mapping

import pytest
import torch
from torch import nn

from breprecover.recovery.decision import (
    AcceptanceReport,
    BackendTimeoutError,
    recover_failed_shape,
)
from breprecover.cli import REQUIRED_ARTIFACTS, validate_config
from breprecover.selection.models import (
    INPUT_KEYS,
    RepairAbstainRouter,
    ShapeVetoRouter,
    build_candidate_input,
    load_state_dict,
    predict_candidate_probabilities,
    predict_shape_probability,
)
from breprecover.selection.shape_evidence import aggregate_shape_evidence


def candidate_inputs(count: int = 4) -> dict[str, torch.Tensor]:
    return {
        "relation_embedding": torch.arange(count * 128, dtype=torch.float32).reshape(count, 128) / 100,
        "route_logits": torch.arange(count * 3, dtype=torch.float32).reshape(count, 3) / 10,
        "compatibility_logits": torch.linspace(-1, 1, count),
        "abstention_logits": torch.linspace(1, -1, count).reshape(count, 1),
        "relation_features": torch.ones(count, 5),
    }


def test_candidate_input_order_and_dimensions() -> None:
    inputs = candidate_inputs()
    actual = build_candidate_input(inputs)
    expected = torch.cat(
        [
            inputs["relation_embedding"],
            inputs["route_logits"],
            inputs["compatibility_logits"].unsqueeze(-1),
            inputs["abstention_logits"],
            inputs["relation_features"],
        ],
        dim=-1,
    )
    assert INPUT_KEYS == tuple(inputs)
    assert actual.shape == (4, 138)
    assert torch.equal(actual, expected)


def test_candidate_model_outputs_finite_candidate_logits() -> None:
    torch.manual_seed(31)
    output = RepairAbstainRouter().eval()(candidate_inputs())
    assert output["repair_logit"].shape == (4,)
    assert output["abstain_logit"].shape == (4,)
    assert all(torch.isfinite(value).all() for value in output.values())


def test_single_candidate_score_is_supported_and_empty_is_rejected() -> None:
    model = RepairAbstainRouter().eval()
    repair, abstain = predict_candidate_probabilities(model, candidate_inputs(1))
    assert repair.shape == abstain.shape == (1,)
    empty = {key: value[:0] for key, value in candidate_inputs(1).items()}
    with pytest.raises(ValueError, match="nonempty"):
        predict_candidate_probabilities(model, empty)


def test_nonfinite_candidate_or_shape_scores_fail_closed() -> None:
    class NonfiniteCandidate(RepairAbstainRouter):
        def forward(self, inputs):
            return {"repair_logit": torch.tensor([float("inf")]), "abstain_logit": torch.tensor([0.0])}

    with pytest.raises(RuntimeError, match="nonfinite logits"):
        predict_candidate_probabilities(NonfiniteCandidate(), candidate_inputs(1))

    class NonfiniteShape(ShapeVetoRouter):
        def forward(self, features):
            return torch.tensor([float("inf")])

    with pytest.raises(RuntimeError, match="nonfinite or malformed logits"):
        predict_shape_probability(NonfiniteShape(), torch.zeros(12))

    bad_inputs = candidate_inputs(1)
    bad_inputs["relation_embedding"][0, 0] = float("inf")
    with pytest.raises(RuntimeError, match="candidate input contains nonfinite"):
        predict_candidate_probabilities(RepairAbstainRouter(), bad_inputs)


def test_shape_model_and_probability_contract() -> None:
    torch.manual_seed(32)
    model = ShapeVetoRouter().eval()
    logits = model(torch.zeros(2, 12))
    assert logits.shape == (2,)
    assert torch.isfinite(logits).all()
    assert ShapeVetoRouter().net[1].out_features == 32


def test_checkpoint_loader_requires_real_checkpoint(tmp_path: Path) -> None:
    model = RepairAbstainRouter()
    absent = tmp_path / "absent.pt"
    with pytest.raises(FileNotFoundError):
        load_state_dict(model, str(absent))
    checkpoint = tmp_path / "router.pt"
    torch.save({"model": model.state_dict()}, checkpoint)
    loaded = load_state_dict(RepairAbstainRouter(), str(checkpoint))
    assert not loaded.training
    assert all(not parameter.requires_grad for parameter in loaded.parameters())

    bare_checkpoint = tmp_path / "router-state.pt"
    torch.save(model.state_dict(), bare_checkpoint)
    bare_loaded = load_state_dict(RepairAbstainRouter(), str(bare_checkpoint))
    for expected, actual in zip(model.parameters(), bare_loaded.parameters()):
        assert torch.equal(expected, actual)


def test_config_cli_validation_checks_external_files(tmp_path: Path) -> None:
    artifacts = {}
    for key in REQUIRED_ARTIFACTS:
        file_path = tmp_path / f"{key}.bin"
        if key == "shape_policy_metadata":
            file_path.write_text(json.dumps({"selected_dev_policy": {"threshold": 0.561}}), encoding="utf-8")
        else:
            file_path.touch()
        artifacts[key] = file_path.name
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"artifacts": artifacts, "policy": {"shape_threshold": 0.561}}), encoding="utf-8")
    result = validate_config(config_path)
    assert result["valid"] is True
    assert result["shape_threshold"] == 0.561
    config_path.write_text(json.dumps({"artifacts": artifacts, "policy": {"shape_threshold": True}}), encoding="utf-8")
    with pytest.raises(ValueError, match="not a bool or string"):
        validate_config(config_path)
    config_path.write_text(json.dumps({"artifacts": artifacts, "policy": {"shape_threshold": 0.561}}), encoding="utf-8")
    metadata_path = tmp_path / artifacts["shape_policy_metadata"]
    metadata_path.write_text(json.dumps({"selected_dev_policy": {"threshold": True}}), encoding="utf-8")
    with pytest.raises(ValueError, match="not a bool or string"):
        validate_config(config_path)
    metadata_path.write_text(json.dumps({"selected_dev_policy": {"threshold": 0.561}}), encoding="utf-8")
    config_path.write_text(json.dumps({"artifacts": artifacts, "policy": {"shape_threshold": 0.57}}), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match frozen metadata"):
        validate_config(config_path)
    config_path.write_text(json.dumps({"artifacts": artifacts, "policy": {"shape_threshold": 0.561}}), encoding="utf-8")
    (tmp_path / artifacts["shape_selector_checkpoint"]).unlink()
    with pytest.raises(FileNotFoundError, match="shape_selector_checkpoint"):
        validate_config(config_path)


class _Backend:
    def __init__(self, candidate: Any = "candidate", error: Exception | None = None):
        self.candidate = candidate
        self.error = error
        self.calls = 0
        self.timeout: float | None = None

    def recover(self, failed_shape: Any, *, timeout_seconds: float) -> Any:
        self.calls += 1
        self.timeout = timeout_seconds
        if self.error:
            raise self.error
        return self.candidate


class _Validator:
    def __init__(self, report: AcceptanceReport | None = None, error: Exception | None = None):
        self.report = report or AcceptanceReport(True, True, True, True)
        self.error = error
        self.calls = 0

    def validate(self, candidate: Any) -> AcceptanceReport:
        self.calls += 1
        if self.error:
            raise self.error
        return self.report


def run(official_valid: bool, score: float, backend: _Backend, validator: _Validator):
    return recover_failed_shape(
        "official",
        official_valid=official_valid,
        shape_score=score,
        backend=backend,
        validator=validator,
        timeout_seconds=3.5,
    )


def test_official_success_bypasses_selector_and_backend() -> None:
    backend, validator = _Backend(), _Validator()
    result = run(True, 0.0, backend, validator)
    assert result.shape == "official"
    assert result.decision == "official_success_no_route"
    assert backend.calls == validator.calls == 0


@pytest.mark.parametrize("score", [0.560999, 0.0])
def test_below_frozen_threshold_does_not_invoke_backend(score: float) -> None:
    backend, validator = _Backend(), _Validator()
    result = run(False, score, backend, validator)
    assert result.decision == "fallback_below_threshold"
    assert result.official_result_preserved and not result.accepted
    assert backend.calls == validator.calls == 0


def test_threshold_equality_routes_and_requires_four_acceptance_checks() -> None:
    backend, validator = _Backend(), _Validator()
    result = run(False, 0.561, backend, validator)
    assert result.shape == "candidate"
    assert result.decision == "recovered_and_accepted"
    assert backend.timeout == 3.5


@pytest.mark.parametrize("score", [-0.01, 1.01])
def test_out_of_range_shape_score_abstains(score: float) -> None:
    backend, validator = _Backend(), _Validator()
    result = run(False, score, backend, validator)
    assert result.decision == "fallback_invalid_score"
    assert backend.calls == validator.calls == 0


@pytest.mark.parametrize(
    "report",
    [
        AcceptanceReport(False, True, True, True),
        AcceptanceReport(True, False, True, True),
        AcceptanceReport(True, True, False, True),
        AcceptanceReport(True, True, True, False),
        AcceptanceReport(True, True, True, True, {"volume_delta": 999.0}),
    ],
)
def test_each_acceptance_gate_is_independent_and_fails_closed(report: AcceptanceReport) -> None:
    backend, validator = _Backend(), _Validator(report)
    result = run(False, 0.561, backend, validator)
    if report.accepted:
        assert result.shape == "candidate"
    else:
        assert result.shape == "official"
        assert result.decision == "fallback_validation_rejected"


@pytest.mark.parametrize("field", ["kernel_valid", "single_solid", "boundary_edges_valid", "step_roundtrip_valid"])
def test_acceptance_report_rejects_truthy_non_bool(field: str) -> None:
    values = {key: True for key in ("kernel_valid", "single_solid", "boundary_edges_valid", "step_roundtrip_valid")}
    values[field] = 1
    with pytest.raises(TypeError, match="built-in bool"):
        AcceptanceReport(**values)


@pytest.mark.parametrize("value", [1, "true", None])
def test_official_validity_requires_a_real_bool(value: Any) -> None:
    with pytest.raises(TypeError, match="built-in bool"):
        recover_failed_shape(
            "official",
            official_valid=value,
            shape_score=0.0,
            backend=_Backend(),
            validator=_Validator(),
            timeout_seconds=1.0,
        )


@pytest.mark.parametrize(
    "error,decision",
    [
        (BackendTimeoutError("budget expired"), "fallback_timeout"),
        (RuntimeError("backend failed"), "fallback_exception"),
    ],
)
def test_backend_timeout_and_exception_preserve_official(error: Exception, decision: str) -> None:
    result = run(False, 0.561, _Backend(error=error), _Validator())
    assert result.shape == "official"
    assert result.decision == decision


def test_validator_exception_preserves_official() -> None:
    result = run(False, 0.561, _Backend(), _Validator(error=RuntimeError("kernel unavailable")))
    assert result.shape == "official"
    assert result.decision == "fallback_exception"


def test_validator_dict_with_missing_acceptance_fields_cannot_pass() -> None:
    class MissingValidator:
        def validate(self, candidate):
            return {"kernel_valid": True, "single_solid": True}

    result = recover_failed_shape(
        "official",
        official_valid=False,
        shape_score=0.561,
        backend=_Backend(),
        validator=MissingValidator(),
        timeout_seconds=2.0,
    )
    assert result.shape == "official"
    assert result.decision == "fallback_exception"
    assert "AcceptanceReport" in result.error


@pytest.mark.parametrize("score", [float("nan"), float("inf")])
def test_nonfinite_shape_score_abstains(score: float) -> None:
    backend, validator = _Backend(), _Validator()
    result = run(False, score, backend, validator)
    assert result.decision == "fallback_nonfinite_score"
    assert backend.calls == validator.calls == 0


def test_optional_parity_with_original_networks_and_aggregation() -> None:
    source_root = os.environ.get("BREPRECOVER_REFERENCE_ROOT")
    if not source_root:
        pytest.skip("set BREPRECOVER_REFERENCE_ROOT for frozen network parity checks")
    root = Path(source_root)
    candidate_tree = ast.parse((root / "boundary/stage278p.py").read_text(encoding="utf-8"))
    input_tree = ast.parse((root / "boundary/stage244r5a.py").read_text(encoding="utf-8"))
    shape_tree = ast.parse((root / "boundary/stage286rp.py").read_text(encoding="utf-8"))

    def node(tree: ast.Module, name: str, kind=ast.FunctionDef):
        return next(item for item in tree.body if isinstance(item, kind) and item.name == name)

    common: dict[str, Any] = {
        "Any": Any,
        "Mapping": Mapping,
        "torch": torch,
        "nn": nn,
    }
    original_input_ns = {
        **common,
        "INPUT_KEYS": INPUT_KEYS,
        "INPUT_DIM": 138,
        "FORBIDDEN_INPUT_KEYS": {"family", "families", "source_id", "source_shape", "case_id", "case_position", "candidate_position", "primary_code"},
    }
    original_input_node = node(input_tree, "build_selector_input")
    exec(compile(ast.Module(body=[original_input_node], type_ignores=[]), "<isolated-selector-input>", "exec"), original_input_ns)
    original_candidate_ns = {
        **common,
        "INPUT_DIM": 138,
        "build_selector_input": original_input_ns["build_selector_input"],
    }
    exec(
        compile(ast.Module(body=[node(candidate_tree, "RepairAbstainRouter", ast.ClassDef)], type_ignores=[]), "<isolated-candidate-router>", "exec"),
        original_candidate_ns,
    )
    original_shape_ns = {**common, "INPUT_DIM": 12}
    exec(
        compile(ast.Module(body=[node(shape_tree, "ShapeVetoRouter", ast.ClassDef)], type_ignores=[]), "<isolated-shape-router>", "exec"),
        original_shape_ns,
    )
    OriginalCandidate = original_candidate_ns["RepairAbstainRouter"]
    OriginalShape = original_shape_ns["ShapeVetoRouter"]

    torch.manual_seed(719)
    original_candidate, actual_candidate = OriginalCandidate().eval(), RepairAbstainRouter().eval()
    actual_candidate.load_state_dict(original_candidate.state_dict(), strict=True)
    inputs = candidate_inputs()
    assert torch.equal(build_candidate_input(inputs), original_input_ns["build_selector_input"](inputs))
    expected_logits = original_candidate(inputs)
    actual_logits = actual_candidate(inputs)
    for key in expected_logits:
        assert torch.equal(actual_logits[key], expected_logits[key])

    repair = expected_logits["repair_logit"].sigmoid()
    abstain = expected_logits["abstain_logit"].sigmoid()
    geometry = torch.tensor([True, False, True, True])
    actual_features = aggregate_shape_evidence(repair, abstain, geometry)
    shape_input = actual_features.unsqueeze(0)
    original_shape, actual_shape = OriginalShape(32).eval(), ShapeVetoRouter(32).eval()
    actual_shape.load_state_dict(original_shape.state_dict(), strict=True)
    assert torch.equal(actual_shape(shape_input), original_shape(shape_input))
