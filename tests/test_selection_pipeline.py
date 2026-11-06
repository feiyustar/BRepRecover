from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from breprecover.selection.models import (
    INPUT_KEYS,
    RepairAbstainRouter,
    ShapeVetoRouter,
    load_state_dict,
    predict_candidate_probabilities,
    predict_shape_probability,
)
from breprecover.selection.pipeline import (
    EVIDENCE_SCHEMA,
    SELECTION_SCHEMA,
    CandidateEvidenceV1,
    LoadedSelector,
    load_selector_from_state_dicts,
    select_recovery_route,
    validate_candidate_evidence,
)
from breprecover.selection.shape_evidence import aggregate_shape_evidence


def make_evidence(count: int = 5) -> CandidateEvidenceV1:
    generator = torch.Generator().manual_seed(6200 + count)
    features = {
        "relation_embedding": torch.randn(count, 128, generator=generator),
        "route_logits": torch.randn(count, 3, generator=generator),
        "compatibility_logits": torch.randn(count),
        "abstention_logits": torch.randn(count, 1),
        "relation_features": torch.randn(count, 5, generator=generator),
    }
    return CandidateEvidenceV1(features, torch.tensor([True, False, True, True, False][:count]))


def make_selector(seed: int = 735) -> LoadedSelector:
    torch.manual_seed(seed)
    candidate = RepairAbstainRouter()
    shape = ShapeVetoRouter()
    return load_selector_from_state_dicts(
        candidate.state_dict(),
        shape.state_dict(),
        configured_threshold=0.561,
        frozen_metadata_threshold=0.561,
    )


def test_evidence_v1_contract_accepts_exact_five_groups() -> None:
    evidence = make_evidence()
    assert evidence.schema_version == EVIDENCE_SCHEMA
    assert set(evidence.features) == set(INPUT_KEYS)
    assert validate_candidate_evidence(evidence) == 5


def test_evidence_rejects_bad_schema_shape_mask_and_nonfinite() -> None:
    evidence = make_evidence()
    with pytest.raises(ValueError, match="schema"):
        validate_candidate_evidence(replace(evidence, schema_version="future.v2"))
    wrong_mask = replace(evidence, geometry_compatible=torch.ones(5))
    with pytest.raises(ValueError, match="bool tensor"):
        validate_candidate_evidence(wrong_mask)
    wrong_count = dict(evidence.features)
    wrong_count["route_logits"] = torch.zeros(4, 3)
    with pytest.raises(ValueError, match="same candidate count"):
        validate_candidate_evidence(replace(evidence, features=wrong_count))
    nonfinite = dict(evidence.features)
    nonfinite["relation_features"] = nonfinite["relation_features"].clone()
    nonfinite["relation_features"][0, 0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        validate_candidate_evidence(replace(evidence, features=nonfinite))


def test_evidence_rejects_mixed_devices_without_gpu() -> None:
    evidence = make_evidence()
    mixed = dict(evidence.features)
    mixed["route_logits"] = torch.empty((5, 3), device="meta")
    with pytest.raises(ValueError, match="share one device"):
        validate_candidate_evidence(replace(evidence, features=mixed))


def test_empty_evidence_fails_closed() -> None:
    empty = CandidateEvidenceV1(
        {
            "relation_embedding": torch.empty(0, 128),
            "route_logits": torch.empty(0, 3),
            "compatibility_logits": torch.empty(0),
            "abstention_logits": torch.empty(0),
            "relation_features": torch.empty(0, 5),
        },
        torch.empty(0, dtype=torch.bool),
    )
    result = select_recovery_route(official_valid=False, evidence=empty, selector=make_selector())
    assert result.decision == "fallback_empty_candidates"
    assert result.official_result_preserved and not result.recovery_eligible


def test_missing_or_random_weights_never_enter_selector(tmp_path) -> None:
    result = select_recovery_route(official_valid=False, evidence=make_evidence(), selector=None)
    assert result.decision == "fallback_selector_unavailable"
    with pytest.raises(ValueError, match="explicitly loaded"):
        LoadedSelector(RepairAbstainRouter(), ShapeVetoRouter(), 0.561)
    loaded = make_selector()
    with pytest.raises(ValueError, match="numeric value"):
        replace(loaded, threshold=True)
    with pytest.raises(ValueError, match="does not match frozen metadata"):
        load_selector_from_state_dicts(
            RepairAbstainRouter().state_dict(),
            ShapeVetoRouter().state_dict(),
            configured_threshold=0.57,
            frozen_metadata_threshold=0.561,
        )
    with pytest.raises(ValueError, match="finite numeric threshold"):
        load_selector_from_state_dicts(
            RepairAbstainRouter().state_dict(),
            ShapeVetoRouter().state_dict(),
            configured_threshold=True,
            frozen_metadata_threshold=True,
        )
    with pytest.raises(FileNotFoundError):
        load_state_dict(RepairAbstainRouter(), str(tmp_path / "missing.pt"))


def test_official_success_bypasses_selection() -> None:
    result = select_recovery_route(official_valid=True, evidence=None, selector=None)
    assert result.decision == "official_success_no_selection"
    assert result.official_result_preserved
    assert result.shape_score is None and not result.candidate_scores


def test_model_and_evidence_device_mismatch_fails_closed() -> None:
    selector = make_selector()
    selector.candidate_model.to("meta")
    result = select_recovery_route(official_valid=False, evidence=make_evidence(), selector=selector)
    assert result.decision == "fallback_invalid_evidence"
    assert "device does not match" in result.failure_reason


def test_hierarchical_api_connects_candidate_aggregate_shape_and_threshold(monkeypatch) -> None:
    selector = make_selector()
    evidence = make_evidence()
    candidate_probs = predict_candidate_probabilities(selector.candidate_model, evidence.features)
    manual_aggregate = aggregate_shape_evidence(*candidate_probs, evidence.geometry_compatible)
    manual_score = predict_shape_probability(selector.shape_model, manual_aggregate)

    result = select_recovery_route(official_valid=False, evidence=evidence, selector=selector)
    assert result.schema_version == SELECTION_SCHEMA
    assert result.shape_features == tuple(manual_aggregate.tolist())
    assert result.shape_score == manual_score
    assert result.recovery_eligible is (manual_score >= selector.threshold)
    assert result.decision == ("eligible_for_recovery" if result.recovery_eligible else "fallback_below_threshold")
    assert len(result.candidate_scores) == 5
    assert result.as_dict()["threshold"] == 0.561


@pytest.mark.parametrize("score,expected", [(0.561, True), (0.560999, False)])
def test_frozen_threshold_uses_inclusive_boundary(monkeypatch, score: float, expected: bool) -> None:
    from breprecover.selection import pipeline

    monkeypatch.setattr(pipeline, "predict_shape_probability", lambda _model, _features: score)
    result = pipeline.select_recovery_route(official_valid=False, evidence=make_evidence(), selector=make_selector())
    assert result.recovery_eligible is expected


def test_all_geometry_ineligible_candidates_are_governed_only_by_shape_gate(monkeypatch) -> None:
    from breprecover.selection import pipeline

    monkeypatch.setattr(pipeline, "predict_shape_probability", lambda _model, _features: 0.561)
    evidence = make_evidence()
    evidence = replace(evidence, geometry_compatible=torch.zeros(5, dtype=torch.bool))
    result = pipeline.select_recovery_route(official_valid=False, evidence=evidence, selector=make_selector())
    assert result.recovery_eligible
    assert result.shape_features[6] == 0.0
