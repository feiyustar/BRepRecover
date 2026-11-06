from __future__ import annotations

import pytest
import torch

from breprecover.recovery.decision import AcceptanceReport, BackendTimeoutError
from breprecover.selection.models import RepairAbstainRouter, ShapeVetoRouter
from breprecover.selection.pipeline import (
    CandidateEvidenceV1,
    load_selector_from_state_dicts,
    run_selector_and_recovery,
)


def selector(threshold: float = 0.561):
    torch.manual_seed(111)
    candidate, shape = RepairAbstainRouter(), ShapeVetoRouter()
    return load_selector_from_state_dicts(
        candidate.state_dict(),
        shape.state_dict(),
        configured_threshold=threshold,
        frozen_metadata_threshold=threshold,
    )


def evidence() -> CandidateEvidenceV1:
    return CandidateEvidenceV1(
        {
            "relation_embedding": torch.zeros(2, 128),
            "route_logits": torch.zeros(2, 3),
            "compatibility_logits": torch.zeros(2),
            "abstention_logits": torch.zeros(2),
            "relation_features": torch.zeros(2, 5),
        },
        torch.tensor([True, False]),
    )


class Adapter:
    def __init__(self, result=evidence(), error=None):
        self.result = result
        self.error = error
        self.calls = 0
        self.observed_shape = None

    def build_candidate_evidence(self, failed_shape):
        self.calls += 1
        self.observed_shape = failed_shape
        if self.error:
            raise self.error
        return self.result


class Backend:
    def __init__(self, result="recovered", error=None):
        self.result = result
        self.error = error
        self.calls = 0
        self.timeout = None

    def recover(self, failed_shape, *, timeout_seconds):
        self.calls += 1
        self.timeout = timeout_seconds
        if self.error:
            raise self.error
        return self.result


class Validator:
    def __init__(self, report=None, error=None):
        self.report = report or AcceptanceReport(True, True, True, True)
        self.error = error
        self.calls = 0

    def validate(self, candidate):
        self.calls += 1
        if self.error:
            raise self.error
        return self.report


def invoke(**kwargs):
    defaults = {
        "official_shape": "original",
        "official_valid": False,
        "adapter": Adapter(),
        "selector": selector(),
        "backend": Backend(),
        "validator": Validator(),
        "timeout_seconds": 8.0,
    }
    defaults.update(kwargs)
    return run_selector_and_recovery(**defaults)


def test_official_success_bypasses_all_injected_components() -> None:
    adapter = Adapter(error=AssertionError("must not run"))
    backend = Backend(error=AssertionError("must not run"))
    validator = Validator(error=AssertionError("must not run"))
    result = invoke(
        official_valid=True,
        adapter=adapter,
        selector=None,
        backend=backend,
        validator=validator,
    )
    assert result.selection is None
    assert result.recovery.shape == "original"
    assert result.recovery.decision == "official_success_no_route"
    assert adapter.calls == backend.calls == validator.calls == 0


def test_low_score_runs_adapter_but_skips_backend_and_validator(monkeypatch) -> None:
    from breprecover.selection import pipeline

    monkeypatch.setattr(pipeline, "predict_shape_probability", lambda *_: 0.560999)
    adapter, backend, validator = Adapter(), Backend(), Validator()
    result = invoke(adapter=adapter, backend=backend, validator=validator)
    assert result.selection.decision == "fallback_below_threshold"
    assert result.recovery.shape == "original"
    assert adapter.calls == 1 and backend.calls == validator.calls == 0


def test_adapter_exception_or_empty_result_fails_closed() -> None:
    adapter_error, backend = Adapter(error=RuntimeError("adapter unavailable")), Backend()
    result = invoke(adapter=adapter_error, backend=backend)
    assert result.selection.decision == "fallback_adapter_exception"
    assert result.recovery.shape == "original"
    assert result.recovery.official_result_preserved
    assert backend.calls == 0

    adapter_empty, backend = Adapter(result=None), Backend()
    result = invoke(adapter=adapter_empty, backend=backend)
    assert result.selection.decision == "fallback_adapter_no_evidence"
    assert result.recovery.shape == "original"
    assert backend.calls == 0

    adapter_no_candidates, backend = Adapter(result=replace_empty_evidence()), Backend()
    result = invoke(adapter=adapter_no_candidates, backend=backend)
    assert result.selection.decision == "fallback_empty_candidates"
    assert result.recovery.shape == "original"
    assert backend.calls == 0


def test_missing_selector_does_not_call_feature_adapter() -> None:
    adapter, backend, validator = Adapter(), Backend(), Validator()
    result = invoke(selector=None, adapter=adapter, backend=backend, validator=validator)
    assert result.selection.decision == "fallback_selector_unavailable"
    assert adapter.calls == backend.calls == validator.calls == 0


def replace_empty_evidence() -> CandidateEvidenceV1:
    return CandidateEvidenceV1(
        {
            "relation_embedding": torch.empty(0, 128),
            "route_logits": torch.empty(0, 3),
            "compatibility_logits": torch.empty(0),
            "abstention_logits": torch.empty(0),
            "relation_features": torch.empty(0, 5),
        },
        torch.empty(0, dtype=torch.bool),
    )


@pytest.mark.parametrize(
    "backend_error,validation_error,report,expected",
    [
        (BackendTimeoutError("expired"), None, None, "fallback_timeout"),
        (RuntimeError("builder"), None, None, "fallback_exception"),
        (None, RuntimeError("validator"), None, "fallback_exception"),
        (None, None, AcceptanceReport(True, False, True, True), "fallback_validation_rejected"),
    ],
)
def test_backend_or_validation_failure_preserves_original(
    monkeypatch, backend_error, validation_error, report, expected
) -> None:
    from breprecover.selection import pipeline

    monkeypatch.setattr(pipeline, "predict_shape_probability", lambda *_: 0.561)
    adapter = Adapter()
    backend = Backend(error=backend_error)
    validator = Validator(report=report, error=validation_error)
    result = invoke(adapter=adapter, backend=backend, validator=validator)
    assert result.selection.recovery_eligible
    assert result.recovery.decision == expected
    assert result.recovery.shape == "original"
    assert backend.calls == 1
    assert validator.calls == (0 if backend_error else 1)


def test_accepted_candidate_uses_same_configured_threshold(monkeypatch) -> None:
    from breprecover.selection import pipeline

    monkeypatch.setattr(pipeline, "predict_shape_probability", lambda *_: 0.7)
    adapter, backend, validator = Adapter(), Backend(), Validator()
    result = invoke(
        selector=selector(0.7),
        adapter=adapter,
        backend=backend,
        validator=validator,
    )
    assert result.selection.threshold == 0.7
    assert result.selection.recovery_eligible
    assert result.recovery.shape == "recovered"
    assert result.recovery.decision == "recovered_and_accepted"
    assert backend.timeout == 8.0
    assert backend.calls == validator.calls == 1
