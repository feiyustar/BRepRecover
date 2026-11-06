"""Exercise selector wiring with deterministic synthetic tensors only."""
from __future__ import annotations

import torch

from breprecover.selection.models import RepairAbstainRouter, ShapeVetoRouter
from breprecover.selection.pipeline import (
    CandidateEvidenceV1,
    load_selector_from_state_dicts,
    select_recovery_route,
)


def main() -> None:
    torch.manual_seed(7)
    candidate_model = RepairAbstainRouter()
    shape_model = ShapeVetoRouter()
    selector = load_selector_from_state_dicts(
        candidate_model.state_dict(),
        shape_model.state_dict(),
        configured_threshold=0.561,
        frozen_metadata_threshold=0.561,
    )
    evidence = CandidateEvidenceV1(
        features={
            "relation_embedding": torch.zeros(2, 128),
            "route_logits": torch.zeros(2, 3),
            "compatibility_logits": torch.zeros(2),
            "abstention_logits": torch.zeros(2),
            "relation_features": torch.zeros(2, 5),
        },
        geometry_compatible=torch.tensor([True, False]),
    )
    result = select_recovery_route(
        official_valid=False, evidence=evidence, selector=selector
    )
    print(result.as_dict())


if __name__ == "__main__":
    main()
