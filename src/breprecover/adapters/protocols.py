"""Structural contracts for external feature adapters.

Implementations must derive candidate evidence from permitted observed CAD
inputs and keep candidate row ordering aligned across all features and masks.
No concrete CAD adapter is included in this package.
"""
from __future__ import annotations

from typing import Any, Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from breprecover.selection.pipeline import CandidateEvidenceV1


class CandidateFeatureAdapter(Protocol):
    def build_candidate_evidence(self, failed_shape: Any) -> CandidateEvidenceV1 | None:
        """Return same-order candidate tensors or None when no evidence exists."""
