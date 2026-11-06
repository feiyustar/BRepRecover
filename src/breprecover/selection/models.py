"""Frozen selector model contracts extracted from the research implementation.

Architecture provenance is based on the current research source and remains
subject to author/rights-holder confirmation; this is not a clean-room claim.
"""
from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

INPUT_KEYS = (
    "relation_embedding",
    "route_logits",
    "compatibility_logits",
    "abstention_logits",
    "relation_features",
)
CANDIDATE_INPUT_DIMS = (128, 3, 1, 1, 5)
CANDIDATE_INPUT_DIM = sum(CANDIDATE_INPUT_DIMS)
SHAPE_INPUT_DIM = 12
SHAPE_HIDDEN_DIM = 32
SHAPE_THRESHOLD = 0.561


def build_candidate_input(inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """Concatenate the exact five Stage278P features into a 138-D tensor."""
    if set(inputs) != set(INPUT_KEYS):
        raise ValueError(f"candidate input keys must be exactly {list(INPUT_KEYS)}")
    relation = inputs["relation_embedding"]
    routes = inputs["route_logits"]
    compatibility = inputs["compatibility_logits"]
    abstention = inputs["abstention_logits"]
    relation_features = inputs["relation_features"]
    if routes.ndim < 2:
        raise ValueError("route_logits must include a candidate and feature dimension")
    if compatibility.ndim == routes.ndim - 1:
        compatibility = compatibility.unsqueeze(-1)
    if abstention.ndim == routes.ndim - 1:
        abstention = abstention.unsqueeze(-1)
    values = (relation, routes, compatibility, abstention, relation_features)
    if any(value.shape[:-1] != routes.shape[:-1] for value in values):
        raise ValueError("candidate feature leading dimensions must match")
    if tuple(value.shape[-1] for value in values) != CANDIDATE_INPUT_DIMS:
        raise ValueError("candidate feature dimensions must be (128, 3, 1, 1, 5)")
    return torch.cat(values, dim=-1).float()


class RepairAbstainRouter(nn.Module):
    """Stage278P-compatible candidate scoring head; checkpoint required."""

    def __init__(self, hidden_dim: int = 64) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.LayerNorm(CANDIDATE_INPUT_DIM),
            nn.Linear(CANDIDATE_INPUT_DIM, hidden_dim),
            nn.GELU(),
        )
        self.repair_head = nn.Linear(hidden_dim, 1)
        self.abstain_head = nn.Linear(hidden_dim, 1)

    def forward(self, inputs: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        hidden = self.encoder(build_candidate_input(inputs))
        return {
            "repair_logit": self.repair_head(hidden).squeeze(-1),
            "abstain_logit": self.abstain_head(hidden).squeeze(-1),
        }


class ShapeVetoRouter(nn.Module):
    """Stage286R-T-compatible 12-D shape scorer; checkpoint required."""

    def __init__(self, hidden_dim: int = SHAPE_HIDDEN_DIM) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(SHAPE_INPUT_DIM),
            nn.Linear(SHAPE_INPUT_DIM, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[-1] != SHAPE_INPUT_DIM:
            raise ValueError(f"shape features must have final dimension {SHAPE_INPUT_DIM}")
        return self.net(features).squeeze(-1)


def load_state_dict(model: nn.Module, checkpoint_path: str) -> nn.Module:
    """Load an explicitly supplied checkpoint, refusing absent/incompatible files."""
    from pathlib import Path

    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"required model checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("model"), dict):
        state = checkpoint["model"]
    elif isinstance(checkpoint, dict) and all(torch.is_tensor(value) for value in checkpoint.values()):
        state = checkpoint
    else:
        state = None
    if not isinstance(state, dict):
        raise ValueError(f"checkpoint has no model state_dict: {path}")
    model.load_state_dict(state, strict=True)
    return model.eval().requires_grad_(False)


def predict_candidate_probabilities(
    model: RepairAbstainRouter,
    inputs: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score a nonempty candidate set; returned probability vectors are CPU float32."""
    features = build_candidate_input(inputs)
    if features.ndim != 2 or features.shape[0] == 0:
        raise ValueError("candidate scoring requires a nonempty [candidate_count, 138] tensor")
    if not bool(torch.isfinite(features).all()):
        raise RuntimeError("candidate input contains nonfinite values")
    model_device = next(model.parameters()).device
    if features.device != model_device:
        raise ValueError(f"candidate tensor device {features.device} does not match model device {model_device}")
    model.eval()
    with torch.no_grad():
        output = model(inputs)
    repair_logits = output["repair_logit"].reshape(-1)
    abstain_logits = output["abstain_logit"].reshape(-1)
    if repair_logits.numel() != features.shape[0] or abstain_logits.numel() != features.shape[0]:
        raise RuntimeError("candidate model output count does not match input candidates")
    if not bool(torch.isfinite(repair_logits).all() and torch.isfinite(abstain_logits).all()):
        raise RuntimeError("candidate model produced nonfinite logits")
    repair = repair_logits.sigmoid().float().cpu()
    abstain = abstain_logits.sigmoid().float().cpu()
    return repair, abstain


def predict_shape_probability(model: ShapeVetoRouter, features: torch.Tensor) -> float:
    """Score one aggregate, moving it to the model device and failing closed."""
    if features.numel() != SHAPE_INPUT_DIM or features.ndim not in (1, 2):
        raise ValueError("shape scoring requires one 12-value aggregate")
    if not bool(torch.isfinite(features).all()):
        raise RuntimeError("shape aggregate contains nonfinite values")
    model_device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        logits = model(features.reshape(1, SHAPE_INPUT_DIM).to(model_device)).reshape(-1)
    if logits.numel() != 1 or not bool(torch.isfinite(logits).all()):
        raise RuntimeError("shape model produced nonfinite or malformed logits")
    probability = logits.sigmoid().float().cpu()
    if probability.numel() != 1 or not bool(torch.isfinite(probability).all()):
        raise RuntimeError("shape model produced a nonfinite or malformed probability")
    return float(probability.item())
