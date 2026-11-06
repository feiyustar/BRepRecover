# BRepRecover

**Status: planning and prototype only; not a usable or validated release.**

BRepRecover studies a failure-only recovery wrapper for direct B-rep generation. The intended workflow leaves generator outputs that pass existing acceptance checks unchanged, uses explicit boundary evidence to prioritize recovery attempts, separates candidate-level evidence from a whole-shape decision, and accepts an alternative only after independent CAD-kernel and STEP validation. A mature Gmsh/OpenCASCADE healing backend is an execution component, not a new healing algorithm.

The paper reports a wrapper-level gain on evaluated cohorts. It does not establish that learned selection outperforms all alternatives, that every invalid shape is recoverable, or that online validity implies geometric fidelity or engineering readiness.

## Current Contents

This directory contains an installable prototype for frozen candidate/shape scoring architectures, a versioned tensor-evidence input, connected hierarchical selection, and failure-only recovery decision contracts. Its CLI validates configured external artifact paths. An optional pythonocc adapter can validate a supplied shape and perform STEP round-trip checks, but is not a recovery backend. There is still no end-to-end generation, online CAD feature extraction, or Gmsh/OCC healing implementation. A valid config does not imply those components can be run.

The project has selected the MIT License for the authorized BRepRecover project code as of 2026-11-06. This does not license datasets, derived data, weights, third-party dependencies, or excluded upstream-derived source. Provenance and redistribution-rights review for included extracted behavior remains pending, so this prototype is not public-release-ready and is not a reproducible implementation of the paper. See [docs/source_audit.md](docs/source_audit.md) and [docs/release_plan.md](docs/release_plan.md).

## Data and Weights

No datasets, feature caches, captures, experiment results, or model weights are included. Upstream dataset references are provided for provenance only:

- ABC dataset: https://archive.nyu.edu/handle/2451/43778
- DeepCAD project and data instructions: https://github.com/ChrisWu1997/DeepCAD

Review dataset-host terms before acquiring data. If approved for local use later, data and derived caches are expected outside version control under data/ or a configured external path; .gitignore excludes common local data and result artifacts. This repository grants no right to redistribute datasets or derived shape data.

The frozen inference chain requires model artifacts not present here: generator checkpoints, candidate router, shape selector, feature-backbone/runtime bundle, and compatible configuration/metadata. Model classes require explicitly supplied compatible weights for inference; random initialization is used only as a test fixture. `load_state_dict` accepts a plain state dict or a `{"model": state_dict}` mapping using PyTorch's `weights_only=True`; historical training checkpoints with unsupported objects may require conversion in a trusted offline environment. No unsafe pickle-loading fallback is provided. The decision API takes already-computed validity/shape scores and injected recovery/validator implementations. Its backend must enforce the supplied timeout (for example in a timed subprocess), clone/isolate the input shape before edits, and use temporary artifacts. The coordinator only decides which object to return; it does not itself provide a CAD-object or filesystem transaction. No backend is included. The config CLI checks paths and threshold metadata only; it does not verify checkpoint compatibility. A code-only checkout cannot reproduce reported metrics or run the complete learned route. No model download command is provided because an authorized public artifact location and its terms have not been established.

The runtime config's `policy.shape_threshold` must match the frozen Stage286R-T result at `artifacts.shape_policy_metadata` (`selected_dev_policy.threshold`). The current frozen value is 0.561; changing it is a protocol change, not a tuning recommendation. Example policy fragment:

```json
{"policy": {"shape_threshold": 0.561}}
```

## Tensor Selector Example

The following is a synthetic smoke example. It makes deterministic random weights in memory solely to exercise the API; these are not trained or paper weights and must not be used as deployment inference. A real caller must load its approved tensor checkpoints and supply evidence from an independently authorized feature adapter.

```python
import torch
from breprecover.selection.models import RepairAbstainRouter, ShapeVetoRouter
from breprecover.selection.pipeline import (
    CandidateEvidenceV1,
    load_selector_from_state_dicts,
    select_recovery_route,
)

torch.manual_seed(7)
candidate = RepairAbstainRouter()
shape = ShapeVetoRouter()
selector = load_selector_from_state_dicts(
    candidate.state_dict(), shape.state_dict(),
    configured_threshold=0.561, frozen_metadata_threshold=0.561,
)
evidence = CandidateEvidenceV1(
    {
        "relation_embedding": torch.zeros(2, 128),
        "route_logits": torch.zeros(2, 3),
        "compatibility_logits": torch.zeros(2),
        "abstention_logits": torch.zeros(2),
        "relation_features": torch.zeros(2, 5),
    },
    geometry_compatible=torch.tensor([True, False]),
)
print(select_recovery_route(
    official_valid=False, evidence=evidence, selector=selector
).as_dict())
```

`CandidateEvidenceV1` is the `breprecover.candidate-evidence.v1` contract: five floating tensors with candidate-leading dimensions `[N,128]`, `[N,3]`, `[N]` or `[N,1]`, `[N]` or `[N,1]`, `[N,5]`, plus a same-device bool mask `[N]`. The mask is required but may be all false; there is no extra “any eligible candidate” gate beyond the frozen shape score threshold. All six inputs must share a device and be finite. Candidate scoring requires the candidate model on that same device; candidate probabilities are explicitly moved to CPU for the fixed aggregate, then the 12-value vector is moved to the shape model's device for scoring. The selector returns candidate scores, the 12-value shape input, shape score, threshold, and eligibility decision. Eligibility only authorizes a later external recovery step; it does not run CAD construction or acceptance checks. The contract records dimensions and value types, not the feature semantics or checkpoint provenance; those must be supplied and audited by the integrating application. See [docs/feature_adapter_contract.md](docs/feature_adapter_contract.md) for source-traced feature meanings and candidate ordering.

For combined control flow, implement `CandidateFeatureAdapter` and call `run_selector_and_recovery(...)` with a `LoadedSelector`, `RecoveryBackend`, and `IndependentValidator`. The coordinator bypasses all components on official success and preserves the original result on adapter, selector, backend, or validation failures. It is interface composition only; no concrete feature adapter or CAD backend is included. See [docs/backend_contract.md](docs/backend_contract.md).

Synthetic in-memory state dicts in the example/tests only exercise model wiring. They do not establish compatibility with the paper's actual checkpoints, reproduce the paper's results, or validate feature semantics.

## Optional OCCT Validation

`breprecover.validation.OCCTShapeValidator` checks an in-memory shape with
pythonocc. `IsolatedSTEPValidator` is a separate file-based API: it accepts an
existing STEP path, launches a worker to read/check it and perform a STEP
round-trip, and enforces a process timeout. It does not serialize arbitrary
OCC shapes and is not wired as a shape-protocol replacement. Both do
validation only, not healing or construction. pythonocc-core is deliberately
not a mandatory package dependency; the worker was tested with synthetic
shapes in pythonocc-core/OCCT 7.5.1. Process isolation is not a security
sandbox, and native/system-level failures may exceed its guarantees. See
[docs/dependency_audit.md](docs/dependency_audit.md) and
[docs/backend_contract.md](docs/backend_contract.md) for boundaries.

## Environment

See [docs/environment.md](docs/environment.md) for separate CPU prototype and optional CAD-runtime setup instructions. Package metadata requires Python >=3.10 and PyTorch >=2.0. The exact local test pins are recorded there, but have not been verified in a fresh clean environment. The optional OCCT adapter was tested in one local runtime; compatibility, STEP interoperability, and redistribution terms elsewhere remain unverified. Unit tests do not validate a recovery backend or engineering readiness.

## Audit and Plan

- docs/source_audit.md: source/baseline comparisons, provenance classifications, and license questions.
- docs/implementation_map.md: proposed module boundaries and current research-code dependencies.
- docs/release_plan.md: staged gates before code or artifacts can be represented as a release.
- docs/dependency_audit.md: local optional OCCT validation probe, dependency metadata observations, and unresolved licensing/safety boundaries.

To exercise the synthetic selector API, install the package and documented CPU test dependencies, then run `python examples/synthetic_selector_smoke.py`. This uses deterministic random state in memory and is not inference. The `breprecover validate-config PATH.json` command checks all six configured external artifact paths and threshold metadata; it does not load files or perform inference. No model-download, feature-extraction, training, or end-to-end generation/recovery command is supplied.

The optional `breprecover validate-step PATH.step --timeout 30` command validates
an existing STEP file in an isolated OCCT worker; it does not repair or rewrite
the file. It requires a separately prepared environment with compatible
`pythonocc-core`/OCCT. See [docs/usage.md](docs/usage.md) for exit codes, report
handling, and runtime requirements. This validator is not a security sandbox
or a complete geometric-quality audit.
