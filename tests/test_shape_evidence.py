from __future__ import annotations

import ast
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pytest
import torch

from breprecover.selection.shape_evidence import aggregate_shape_evidence


class _FixedRouter(torch.nn.Module):
    def forward(self, inputs: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {
            "repair_logit": inputs["repair_logit"].reshape(-1),
            "abstain_logit": inputs["abstain_logit"].reshape(-1),
        }


def _research_aggregate():
    source_root = os.environ.get("BREPRECOVER_REFERENCE_ROOT")
    if not source_root:
        pytest.skip("set BREPRECOVER_REFERENCE_ROOT to run read-only research-source parity checks")
    path = Path(source_root) / "boundary" / "stage286rp.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_features"
    )
    isolated = ast.Module(body=[function], type_ignores=[])
    namespace: dict[str, Any] = {
        "Any": Any,
        "Mapping": Mapping,
        "nn": torch.nn,
        "torch": torch,
        "INPUT_KEYS": ("repair_logit", "abstain_logit"),
    }
    exec(compile(isolated, str(path), "exec"), namespace)
    return namespace["_features"]


def _parity_case(
    repair_logits: torch.Tensor,
    abstain_logits: torch.Tensor,
    geometry: torch.Tensor,
) -> None:
    old_features = _research_aggregate()
    count = int(repair_logits.numel())
    values = {
        "repair_logit": repair_logits.reshape(-1, 1),
        "abstain_logit": abstain_logits.reshape(-1, 1),
        "geometry_compatible": geometry.reshape(-1).bool(),
        "candidate_case": torch.zeros(count, dtype=torch.long),
        "safe_beneficial": torch.ones(count, dtype=torch.bool),
    }
    indexes = torch.arange(count)
    expected, _ = old_features(_FixedRouter(), values, indexes, 0)
    actual = aggregate_shape_evidence(
        repair_logits.sigmoid(),
        abstain_logits.sigmoid(),
        geometry,
    )
    assert actual.dtype == torch.float32
    assert actual.device.type == "cpu"
    assert actual.shape == (12,)
    assert torch.equal(actual, expected)


def test_single_candidate_matches_research_feature_order() -> None:
    _parity_case(
        torch.tensor([1.1]),
        torch.tensor([-0.9]),
        torch.tensor([True]),
    )


@pytest.mark.parametrize("count", [2, 3, 17, 64])
def test_deterministic_random_inputs_match_research(count: int) -> None:
    rng = np.random.default_rng(9400 + count)
    repair_logits = torch.tensor(rng.normal(size=count), dtype=torch.float32)
    abstain_logits = torch.tensor(rng.normal(size=count), dtype=torch.float32)
    geometry = torch.tensor(rng.integers(0, 2, size=count), dtype=torch.bool)
    _parity_case(repair_logits, abstain_logits, geometry)


@pytest.mark.parametrize(
    "geometry",
    [
        torch.tensor([True, True, True, True]),
        torch.tensor([False, False, False, False]),
    ],
)
def test_all_and_no_geometry_eligible_match_research(geometry: torch.Tensor) -> None:
    _parity_case(
        torch.tensor([2.0, 1.2, 0.2, -1.0]),
        torch.tensor([-2.0, -1.5, -0.8, 0.5]),
        geometry,
    )


def test_empty_candidate_contract_matches_research_failure() -> None:
    with pytest.raises(RuntimeError):
        aggregate_shape_evidence(
            torch.empty(0),
            torch.empty(0),
            torch.empty(0, dtype=torch.bool),
        )


def test_hand_computed_feature_vector() -> None:
    actual = aggregate_shape_evidence(
        torch.tensor([0.9, 0.8, 0.4, 0.5]),
        torch.tensor([0.1, 0.5, 0.3, 0.5]),
        torch.tensor([True, True, False, True]),
    )
    expected = torch.tensor(
        [0.9, 0.65, 0.1, 0.35, 0.8, 0.3, 0.75, 4, 1, 1, 0.25, 0.25],
        dtype=torch.float32,
    )
    assert torch.allclose(actual, expected, rtol=0, atol=1e-7)


def test_routing_threshold_boundaries_are_exact() -> None:
    actual = aggregate_shape_evidence(
        torch.tensor([0.5, 0.75, 0.9, 0.9]),
        torch.tensor([0.49, 0.49, 0.5, 0.1]),
        torch.tensor([True, True, True, False]),
    )
    assert actual[8:12].tolist() == [2.0, 1.0, 0.5, 0.25]


def test_mismatched_candidate_lengths_raise_value_error() -> None:
    with pytest.raises(ValueError, match="equal flattened lengths"):
        aggregate_shape_evidence(
            torch.tensor([0.7, 0.8]),
            torch.tensor([0.2]),
            torch.tensor([True, False]),
        )


def test_stage288_deployment_aggregate_matches_public_helper(monkeypatch) -> None:
    source_root = os.environ.get("BREPRECOVER_REFERENCE_ROOT")
    if not source_root:
        pytest.skip("set BREPRECOVER_REFERENCE_ROOT for Stage288 deployment-source parity")
    source_root = Path(source_root)
    path = source_root / "inference" / "stage288i_mainline.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_shape_features"
    )
    namespace: dict[str, Any] = {
        "Any": Any,
        "Mapping": Mapping,
        "torch": torch,
        "_state_hash": lambda _: "stable-hash",
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)

    class Candidate:
        def __call__(self, _features):
            return {
                "repair_logit": torch.logit(torch.tensor([0.9, 0.8, 0.4, 0.5])),
                "abstain_logit": torch.logit(torch.tensor([0.1, 0.5, 0.3, 0.5])),
            }

    repair = torch.tensor([0.9, 0.8, 0.4, 0.5])
    abstain = torch.tensor([0.1, 0.5, 0.3, 0.5])
    geometry = torch.tensor([True, True, False, True])

    import sys
    import types
    inference_module = types.ModuleType("inference")
    inference_module.__path__ = []
    capture_module = types.ModuleType("inference.stage246i_mainline")
    capture_module.capture_r5c_features = lambda *args, **kwargs: (
        {"dummy": torch.zeros((4, 1))}, [], geometry, {"stage199_forward_calls": 1}
    )
    monkeypatch.setitem(sys.modules, "inference", inference_module)
    monkeypatch.setitem(sys.modules, "inference.stage246i_mainline", capture_module)

    actual, evidence, metadata = namespace["_shape_features"](
        {}, Candidate(), {"model": object(), "config": {}}
    )
    expected = aggregate_shape_evidence(repair, abstain, geometry)
    assert torch.allclose(actual, expected, rtol=0, atol=1e-6)
    assert len(evidence) == 4
    assert metadata["stage199_state_hash_stable"] is True
