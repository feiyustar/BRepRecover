# Feature Adapter Contract

This document specifies the external adapter boundary for
`breprecover.selection.pipeline`. The package does not create these features
from CAD geometry. Semantics below are traced to research-source functions;
they are not a claim of checkpoint or feature-provenance verification.

## Version 1 Input

Implement `CandidateFeatureAdapter.build_candidate_evidence(failed_shape)` and
return `CandidateEvidenceV1` with schema `breprecover.candidate-evidence.v1`.
For N candidate rows, preserve one common ordering across each field and the
mask:

| Field | Shape | Source meaning |
|---|---:|---|
| `relation_embedding` | `[N, 128]` | Stage197 `relation_encoder` latent, captured by a forward hook in Stage246. It is model/checkpoint-specific. |
| `route_logits` | `[N, 3]` | Raw Stage197 route-head logits. Class index order: accept=0, repair=1, abstain=2 (`ROUTE_ACCEPT`, `ROUTE_REPAIR`, `ROUTE_ABSTAIN`). |
| `compatibility_logits` | `[N]` or `[N, 1]` | Raw Stage197 compatibility-head output. |
| `abstention_logits` | `[N]` or `[N, 1]` | Raw Stage197 abstention-head output. |
| `relation_features` | `[N, 5]` | Stage221 values, in order: `surface_mean_ratio`, `surface_max_ratio`, `endpoint_face_mean_ratio`, `endpoint_face_max_ratio`, `endpoint_pair_ratio`. |
| `geometry_compatible` | bool `[N]` | Hard endpoint/successor predicate for those same candidate rows. |

Each of the five feature groups must be a finite floating tensor with the same
N and device. The mask must be a bool tensor `[N]` on that device. N must be
positive. Do not sort, filter, pad, or truncate any group separately. V1 does
not carry stable candidate IDs; ordering mismatch cannot be repaired later.
An all-false geometry mask is valid input and is not an additional gate beyond
the frozen shape score threshold.

## Feature Semantics And Sampling Scope

Research references:

- `boundary/stage197p.py`: `RELATION_FEATURES`, `ROUTE_*` constants, relation
  encoder, and compatibility/route/abstention heads.
- `boundary/stage221o.py`: `_features`, `_target_coedge_id`,
  `build_online_candidates`, `inference_batch`, and `hard_geometry_compatible`.
- `inference/stage246i_mainline.py`: `capture_r5c_features`.
- `boundary/stage244r5a.py`: `build_selector_input`.
- `inference/stage288i_mainline.py`: `_shape_features`.

Stage246 validates widths `(128, 3, 1, 1, 5)` and verifies equal candidate
counts and finite values before returning the captured tuple. Stage221 computes the five ratios from candidate curve samples, sampled face
surface points, face vertices, and observed coedge endpoints. For each sampled
candidate curve point it finds the nearest sampled face-surface point;
`surface_mean_ratio` and `surface_max_ratio` summarize those distances. For
each of the two candidate curve endpoints it finds the nearest vertex on the
face; the mean/max of those two distances give the endpoint-face fields.
`endpoint_pair_ratio` is the smaller forward or reverse mean distance between
candidate endpoints and the observed coedge endpoints. Each distance is
divided by the shape bounding-box diagonal, floored at `1e-12`.
Stage197 applies `log1p(clamp(features, 0, 10))` when constructing its relation
latent. Stage246 exposes the raw five-value tensor in `relation_features`;
the 138-D candidate router concatenates its five groups in the table order and
casts to float, without additional normalization.

`hard_geometry_compatible` requires a two-vertex edge, a found successor,
oriented edge endpoints equal to the observed coedge's start/end vertex IDs,
and the directed end vertex equal to the successor's start vertex ID.

Candidate rows in Stage221 are deterministic from loop/coedge order and local
candidate-tuple enumeration. Stage246 constructs all five feature groups and
the hard-geometry mask from this same row list. The source path selects the
first nonempty loop after sorting loop IDs, then builds candidate rows for
that loop's coedges in its `coedge_ids` order; it does not cover other loops
or guarantee full shape-boundary coverage. Mainline capture passes
`max_candidate_edges=16`; Stage221 uses at most 8 local edge candidates per
coedge and at most 16 candidate tuples per coedge. This is a limited local
proposal scope, not complete boundary enumeration.

Dimensions alone do not establish semantic equivalence, scale/normalization
compatibility, model-state compatibility, or checkpoint provenance. A
different relation encoder, route-class order, or geometric sampling scheme
can yield shape-compatible but invalid input. This version is an input
structure contract, not a feature-semantic attestation or checkpoint
signature.

## Implemented Boundary

Code defines only the `CandidateFeatureAdapter` Protocol and the v1 tensor
validator. There is no generator integration, Stage197 runtime, Stage221
candidate builder, CAD parser, or concrete feature adapter. Adapters must be
independently reviewed for source rights, data handling, row ordering, units,
normalization, mask meaning, checkpoint binding, and fail-closed behavior.
