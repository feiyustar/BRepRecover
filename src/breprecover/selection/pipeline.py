"""Versioned tensor-evidence adapter and connected hierarchical selector.

This consumes already-computed boundary features; it does not extract CAD
features or validate/recover geometry. Model architectures and aggregation
behavior are extracted from the research implementation, with provenance
confirmation still pending.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
import numbers
from typing import Any

import torch
from torch import nn

from breprecover.adapters.protocols import CandidateFeatureAdapter
from breprecover.recovery.decision import (
    AcceptanceReport,
    RecoveryBackend,
    RecoveryResult,
    IndependentValidator,
    recover_failed_shape,
)
from breprecover.selection.models import (
    CANDIDATE_INPUT_DIMS,
    INPUT_KEYS,
    RepairAbstainRouter,
    SHAPE_THRESHOLD,
    ShapeVetoRouter,
    load_state_dict,
    predict_candidate_probabilities,
    predict_shape_probability,
)
from breprecover.selection.shape_evidence import aggregate_shape_evidence

EVIDENCE_SCHEMA = "breprecover.candidate-evidence.v1"
SELECTION_SCHEMA = "breprecover.hierarchical-selection.v1"


@dataclass(frozen=True)
class CandidateEvidenceV1:
    """Precomputed per-candidate features from an external compliant adapter."""

    features: Mapping[str, torch.Tensor]
    geometry_compatible: torch.Tensor
    schema_version: str = EVIDENCE_SCHEMA


@dataclass(frozen=True)
class LoadedSelector:
    """Two strictly loaded scoring modules bound to one frozen threshold."""

    candidate_model: RepairAbstainRouter
    shape_model: ShapeVetoRouter
    threshold: float

    def __post_init__(self) -> None:
        if getattr(self.candidate_model, "_breprecover_weights_loaded", False) is not True:
            raise ValueError("candidate model must be explicitly loaded from tensor weights")
        if getattr(self.shape_model, "_breprecover_weights_loaded", False) is not True:
            raise ValueError("shape model must be explicitly loaded from tensor weights")
        if isinstance(self.threshold, bool) or not isinstance(self.threshold, numbers.Real):
            raise ValueError("selector threshold must be a finite numeric value")
        if not math.isfinite(self.threshold) or not 0.0 <= self.threshold <= 1.0:
            raise ValueError("selector threshold must be finite and in [0, 1]")


@dataclass(frozen=True)
class SelectionResult:
    schema_version: str
    decision: str
    official_result_preserved: bool
    recovery_eligible: bool
    threshold: float
    candidate_scores: tuple[dict[str, Any], ...] = ()
    shape_features: tuple[float, ...] | None = None
    shape_score: float | None = None
    failure_reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "decision": self.decision,
            "official_result_preserved": self.official_result_preserved,
            "recovery_eligible": self.recovery_eligible,
            "threshold": self.threshold,
            "candidate_scores": list(self.candidate_scores),
            "shape_features": list(self.shape_features) if self.shape_features is not None else None,
            "shape_score": self.shape_score,
            "failure_reason": self.failure_reason,
        }


@dataclass(frozen=True)
class SelectorRecoveryResult:
    selection: SelectionResult | None
    recovery: RecoveryResult


def _load_tensor_state(model: nn.Module, state: Mapping[str, Any], name: str) -> nn.Module:
    if not isinstance(state, Mapping) or not state or not all(torch.is_tensor(value) for value in state.values()):
        raise ValueError(f"{name} state must be a nonempty mapping of tensors")
    model.load_state_dict(dict(state), strict=True)
    model.eval().requires_grad_(False)
    model._breprecover_weights_loaded = True
    return model


def load_selector_from_state_dicts(
    candidate_state: Mapping[str, torch.Tensor],
    shape_state: Mapping[str, torch.Tensor],
    *,
    configured_threshold: float,
    frozen_metadata_threshold: float,
) -> LoadedSelector:
    """Create an inference bundle from explicit pure-tensor state mappings.

    Threshold agreement is checked numerically only; this does not authenticate
    the provenance or integrity of checkpoints or metadata.
    """
    for name, value in (
        ("configured_threshold", configured_threshold),
        ("frozen_metadata_threshold", frozen_metadata_threshold),
    ):
        if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(value):
            raise ValueError(f"{name} must be a finite numeric threshold")
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")
    threshold = float(configured_threshold)
    if abs(threshold - float(frozen_metadata_threshold)) > 1e-9:
        raise ValueError("configured threshold does not match frozen metadata threshold")
    candidate = _load_tensor_state(RepairAbstainRouter(), candidate_state, "candidate")
    shape = _load_tensor_state(ShapeVetoRouter(), shape_state, "shape")
    return LoadedSelector(candidate, shape, threshold)


def load_selector_from_checkpoints(
    candidate_checkpoint: str,
    shape_checkpoint: str,
    *,
    configured_threshold: float,
    frozen_metadata_threshold: float,
) -> LoadedSelector:
    """Load compatible external files through the safe PyTorch loader."""
    for name, value in (
        ("configured_threshold", configured_threshold),
        ("frozen_metadata_threshold", frozen_metadata_threshold),
    ):
        if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(value):
            raise ValueError(f"{name} must be a finite numeric threshold")
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")
    threshold = float(configured_threshold)
    if abs(threshold - float(frozen_metadata_threshold)) > 1e-9:
        raise ValueError("configured threshold does not match frozen metadata threshold")
    candidate = load_state_dict(RepairAbstainRouter(), candidate_checkpoint)
    shape = load_state_dict(ShapeVetoRouter(), shape_checkpoint)
    candidate._breprecover_weights_loaded = True
    shape._breprecover_weights_loaded = True
    return LoadedSelector(candidate, shape, threshold)


def validate_candidate_evidence(evidence: CandidateEvidenceV1) -> int:
    """Validate the v1 schema, dimensions, common device, mask, and values."""
    if not isinstance(evidence, CandidateEvidenceV1) or evidence.schema_version != EVIDENCE_SCHEMA:
        raise ValueError(f"candidate evidence schema must be {EVIDENCE_SCHEMA}")
    if not isinstance(evidence.features, Mapping) or set(evidence.features) != set(INPUT_KEYS):
        raise ValueError(f"candidate evidence keys must be exactly {list(INPUT_KEYS)}")
    values = [evidence.features[key] for key in INPUT_KEYS]
    if not all(torch.is_tensor(value) for value in values):
        raise TypeError("candidate features must all be tensors")
    relation, routes, compatibility, abstention, relation_features = values
    if relation.ndim != 2 or routes.ndim != 2 or relation_features.ndim != 2:
        raise ValueError("relation_embedding, route_logits, relation_features must be rank-2")
    count = int(relation.shape[0])
    if count == 0:
        raise ValueError("candidate evidence is empty")
    if routes.shape[0] != count or relation_features.shape[0] != count:
        raise ValueError("all candidate feature groups must have the same candidate count")
    if tuple(value.shape[1] for value in (relation, routes, relation_features)) != (128, 3, 5):
        raise ValueError("candidate dimensions must be relation=128, route=3, relation_features=5")
    if compatibility.shape not in ((count,), (count, 1)) or abstention.shape not in ((count,), (count, 1)):
        raise ValueError("compatibility_logits and abstention_logits must have shape [N] or [N, 1]")
    if any(not value.is_floating_point() for value in values):
        raise TypeError("all five candidate feature groups must use floating-point tensors")
    device = relation.device
    if any(value.device != device for value in values):
        raise ValueError("candidate feature tensors must share one device")
    mask = evidence.geometry_compatible
    if not torch.is_tensor(mask) or mask.dtype is not torch.bool or tuple(mask.shape) != (count,):
        raise ValueError("geometry_compatible must be a bool tensor with shape [N]")
    if mask.device != device:
        raise ValueError("geometry mask and candidate features must share one device")
    if not all(bool(torch.isfinite(value).all()) for value in values):
        raise ValueError("candidate evidence contains nonfinite values")
    return count


def select_recovery_route(
    *,
    official_valid: bool,
    evidence: CandidateEvidenceV1 | None,
    selector: LoadedSelector | None,
) -> SelectionResult:
    """Run candidate scorer -> 12-D aggregate -> shape scorer -> frozen gate.

    A valid official result bypasses learned selection. All malformed evidence,
    missing weights, empty candidates, and nonfinite model output fail closed.
    This only marks a failed shape eligible for an external recovery backend.
    """
    if type(official_valid) is not bool:
        threshold = selector.threshold if isinstance(selector, LoadedSelector) else SHAPE_THRESHOLD
        return SelectionResult(SELECTION_SCHEMA, "fallback_invalid_official_status", True, False, threshold)
    if official_valid:
        threshold = selector.threshold if isinstance(selector, LoadedSelector) else SHAPE_THRESHOLD
        return SelectionResult(SELECTION_SCHEMA, "official_success_no_selection", True, False, threshold)
    if not isinstance(selector, LoadedSelector):
        return SelectionResult(SELECTION_SCHEMA, "fallback_selector_unavailable", True, False, SHAPE_THRESHOLD, failure_reason="explicitly loaded selector required")
    threshold = selector.threshold
    try:
        count = validate_candidate_evidence(evidence)
        model_device = next(selector.candidate_model.parameters()).device
        if any(value.device != model_device for value in evidence.features.values()):
            raise ValueError("candidate evidence device does not match candidate model device")
        repair, abstain = predict_candidate_probabilities(selector.candidate_model, evidence.features)
        geometry = evidence.geometry_compatible.detach().cpu()
        shape_features = aggregate_shape_evidence(repair, abstain, geometry)
        shape_score = predict_shape_probability(selector.shape_model, shape_features)
        if not math.isfinite(shape_score) or not 0.0 <= shape_score <= 1.0:
            raise RuntimeError("shape score is nonfinite or outside [0, 1]")
        scores = tuple(
            {
                "candidate_position": index,
                "repair_probability": float(repair[index]),
                "abstention_probability": float(abstain[index]),
                "geometry_compatible": bool(geometry[index]),
            }
            for index in range(count)
        )
        features_tuple = tuple(float(value) for value in shape_features.tolist())
        eligible = shape_score >= threshold
        return SelectionResult(
            SELECTION_SCHEMA,
            "eligible_for_recovery" if eligible else "fallback_below_threshold",
            True,
            eligible,
            threshold,
            scores,
            features_tuple,
            shape_score,
        )
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        decision = "fallback_empty_candidates" if isinstance(exc, ValueError) and "empty" in str(exc) else "fallback_invalid_evidence"
        return SelectionResult(SELECTION_SCHEMA, decision, True, False, threshold, failure_reason=reason)


def run_selector_and_recovery(
    official_shape: Any,
    *,
    official_valid: bool,
    adapter: CandidateFeatureAdapter,
    selector: LoadedSelector | None,
    backend: RecoveryBackend,
    validator: IndependentValidator,
    timeout_seconds: float,
) -> SelectorRecoveryResult:
    """Compose feature adapter, loaded selector, and injected recovery checks.

    Official success bypasses adapter, selector, backend, and validator. All
    pre-recovery failures preserve the original value and do not call backend.
    This is control flow only; it provides no shape clone or file transaction.
    """
    if type(official_valid) is not bool:
        return SelectorRecoveryResult(
            None,
            RecoveryResult(official_shape, "fallback_invalid_official_status", True, False),
        )
    if official_valid:
        result = recover_failed_shape(
            official_shape,
            official_valid=True,
            shape_score=0.0,
            backend=backend,
            validator=validator,
            timeout_seconds=timeout_seconds,
            threshold=selector.threshold if isinstance(selector, LoadedSelector) else SHAPE_THRESHOLD,
        )
        return SelectorRecoveryResult(None, result)
    if not isinstance(selector, LoadedSelector):
        selection = select_recovery_route(official_valid=False, evidence=None, selector=None)
        return SelectorRecoveryResult(
            selection,
            RecoveryResult(official_shape, selection.decision, True, False, error=selection.failure_reason),
        )
    try:
        evidence = adapter.build_candidate_evidence(official_shape)
    except Exception as exc:
        selection = SelectionResult(
            SELECTION_SCHEMA,
            "fallback_adapter_exception",
            True,
            False,
            selector.threshold,
            failure_reason=f"{type(exc).__name__}: {exc}",
        )
        return SelectorRecoveryResult(
            selection,
            RecoveryResult(official_shape, selection.decision, True, False, error=selection.failure_reason),
        )
    if evidence is None:
        selection = SelectionResult(
            SELECTION_SCHEMA,
            "fallback_adapter_no_evidence",
            True,
            False,
            selector.threshold,
            failure_reason="feature adapter returned no evidence",
        )
        return SelectorRecoveryResult(
            selection,
            RecoveryResult(official_shape, selection.decision, True, False, error=selection.failure_reason),
        )
    selection = select_recovery_route(official_valid=False, evidence=evidence, selector=selector)
    if not selection.recovery_eligible or selection.shape_score is None:
        return SelectorRecoveryResult(
            selection,
            RecoveryResult(official_shape, selection.decision, True, False, error=selection.failure_reason),
        )
    recovery = recover_failed_shape(
        official_shape,
        official_valid=False,
        shape_score=selection.shape_score,
        backend=backend,
        validator=validator,
        timeout_seconds=timeout_seconds,
        threshold=selector.threshold,
    )
    return SelectorRecoveryResult(selection, recovery)
