"""Shape-level aggregate features used by the frozen recovery selector.

Provenance: behavior extracted from the current DTGBrepGen research
implementation. This is not a clean-room or licensing determination.
"""
from __future__ import annotations

import torch


def aggregate_shape_evidence(
    repair_probability: torch.Tensor,
    abstention_probability: torch.Tensor,
    geometry_compatible: torch.Tensor,
) -> torch.Tensor:
    """Return the frozen 12-value selector input for one candidate set.

    Inputs are flattened per-candidate probabilities and a boolean
    geometry-eligibility mask; all three flattened inputs must have equal
    lengths. The returned CPU float32 tensor preserves the research feature
    order. Empty inputs retain PyTorch's max/min failure behavior; callers
    should fail closed when no candidates are available.
    """
    repair = repair_probability.reshape(-1).float()
    abstain = abstention_probability.reshape(-1).float()
    geometry = geometry_compatible.reshape(-1).bool()
    if repair.numel() != abstain.numel() or repair.numel() != geometry.numel():
        raise ValueError("repair, abstention, and geometry inputs must have equal flattened lengths")
    margin = repair - abstain
    count = max(1, int(repair.numel()))
    routed50 = ((repair >= 0.5) & (abstain < 0.5) & geometry).float()
    routed75 = ((repair >= 0.75) & (abstain < 0.5) & geometry).float()
    return torch.tensor(
        [
            float(repair.max()),
            float(repair.mean()),
            float(abstain.min()),
            float(abstain.mean()),
            float(margin.max()),
            float(margin.mean()),
            float(geometry.sum()) / count,
            float(count),
            float(routed50.sum()),
            float(routed75.sum()),
            float(routed50.mean()),
            float(routed75.mean()),
        ],
        dtype=torch.float32,
    )
