"""Failure-only recovery orchestration with injected construction and validation.

The backend must enforce the supplied execution budget (for example with a
subprocess timeout) and raise BackendTimeoutError on expiry. This module does
not claim to supply a Gmsh/OpenCASCADE backend.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import numbers
from typing import Any, Protocol

from breprecover.selection.models import SHAPE_THRESHOLD


class BackendTimeoutError(TimeoutError):
    """Raised by an injected backend when its enforced time budget expires."""


@dataclass(frozen=True)
class AcceptanceReport:
    kernel_valid: bool
    single_solid: bool
    boundary_edges_valid: bool
    step_roundtrip_valid: bool
    posthoc_properties: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        names = ("kernel_valid", "single_solid", "boundary_edges_valid", "step_roundtrip_valid")
        if any(type(getattr(self, name)) is not bool for name in names):
            raise TypeError("acceptance predicates must be built-in bool values")

    @property
    def accepted(self) -> bool:
        return all(getattr(self, name) is True for name in (
            "kernel_valid", "single_solid", "boundary_edges_valid", "step_roundtrip_valid"
        ))


@dataclass(frozen=True)
class RecoveryResult:
    shape: Any
    decision: str
    official_result_preserved: bool
    accepted: bool
    acceptance: AcceptanceReport | None = None
    error: str | None = None


class RecoveryBackend(Protocol):
    def recover(self, failed_shape: Any, *, timeout_seconds: float) -> Any:
        """Construct from a read-only input or raise; implementation enforces timeout.

        Backends must clone/isolate the official shape before edits and use
        temporary output artifacts. The coordinator provides no CAD-object or
        filesystem transaction.
        """


class IndependentValidator(Protocol):
    def validate(self, candidate: Any) -> AcceptanceReport:
        """Run independent kernel, topology/edge, and STEP checks."""


def recover_failed_shape(
    official_shape: Any,
    *,
    official_valid: bool,
    shape_score: float,
    backend: RecoveryBackend,
    validator: IndependentValidator,
    timeout_seconds: float,
    threshold: float = SHAPE_THRESHOLD,
) -> RecoveryResult:
    """Preserve official success; otherwise recover only above the frozen gate.

    Backend/validator exceptions and explicit timeout failures abstain and
    return the official object. Post-hoc properties are carried for reporting,
    never used as acceptance gates.
    """
    if type(official_valid) is not bool:
        raise TypeError("official_valid must be a built-in bool")
    if official_valid:
        return RecoveryResult(official_shape, "official_success_no_route", True, True)
    if isinstance(threshold, bool) or not isinstance(threshold, numbers.Real):
        raise ValueError("threshold must be a finite number in [0, 1]")
    threshold = float(threshold)
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be a finite number in [0, 1]")
    if isinstance(shape_score, bool) or not isinstance(shape_score, numbers.Real):
        return RecoveryResult(official_shape, "fallback_invalid_score", True, False)
    shape_score = float(shape_score)
    if not math.isfinite(shape_score):
        return RecoveryResult(official_shape, "fallback_nonfinite_score", True, False)
    if not 0.0 <= shape_score <= 1.0:
        return RecoveryResult(official_shape, "fallback_invalid_score", True, False)
    if shape_score < threshold:
        return RecoveryResult(official_shape, "fallback_below_threshold", True, False)
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be a positive finite value")
    try:
        candidate = backend.recover(official_shape, timeout_seconds=timeout_seconds)
        if candidate is None:
            return RecoveryResult(official_shape, "fallback_no_candidate", True, False)
        checks = validator.validate(candidate)
        if not isinstance(checks, AcceptanceReport):
            raise TypeError("validator must return AcceptanceReport")
        if checks.accepted:
            return RecoveryResult(candidate, "recovered_and_accepted", False, True, checks)
        return RecoveryResult(official_shape, "fallback_validation_rejected", True, False, checks)
    except BackendTimeoutError as exc:
        return RecoveryResult(official_shape, "fallback_timeout", True, False, error=str(exc))
    except Exception as exc:
        return RecoveryResult(
            official_shape,
            "fallback_exception",
            True,
            False,
            error=f"{type(exc).__name__}: {exc}",
        )
