# Implementation Map

This map distinguishes the implemented prototype from work that remains. Extracted architecture behavior is based on the current research implementation and subject to author/rights-holder confirmation; absence from a baseline is not treated as permission.

| Package area | Intended responsibility | Research implementation to consult | Dependencies and constraints |
|---|---|---|---|
| evidence/ | Build directed boundary/loop/coedge candidates and local geometric/p-curve evidence; apply hard local eligibility. | boundary/stage221o.py, inference/stage246i_mainline.py capture_r5c_features, inference/stage236_shadow.py. | Current path invokes the Stage199 model (boundary/stage199t.py, stage197p.py) and boundary graph tensorizers. Online input must be defined independently of cached labels. |
| selection/ | Versioned precomputed evidence input, candidate scorer, 12-value aggregate, shape scorer, and frozen threshold. | boundary/stage278p.py, boundary/stage244r5a.py, boundary/stage286rp.py, inference/stage288i_mainline.py. | `pipeline.py` connects v1 evidence -> candidate probabilities -> aggregate -> shape score -> threshold decision. It validates dimensions, boolean geometry mask, finite values, and device alignment. Candidate scores/aggregate cross explicitly to CPU; aggregate moves to the shape model's device. Explicit weights are required. No CAD feature extraction or checkpoint provenance authentication is implemented. |
| recovery/ | Failure-only orchestration with injected backend and independent validator interfaces. | Stage288I/Stage294X routing and inference/gmsh_recovery_sidecar.py. | The prototype strictly preserves declared official success, checks finite [0,1] shape score, gates on score, and requires four built-in-bool acceptance predicates. It does not implement geometry construction or CAD validation. Backend must enforce timeout, clone inputs, and isolate temporary artifacts; coordinator is not itself transactional. |
| validation/ | Independently check CAD-kernel validity, one-solid structure, boundary-edge conditions, and STEP round-trip. | Sidecar _strict, inference/brep_diagnostics.py. | No concrete validator is included. OCC/PythonOCC and STEP read/write APIs require a separately audited adapter. |
| pipeline/ | Artifact config check and thin composition of external adapter, selector, backend, validator. | Research flow references in inference/stage288i_mainline.py, stage294x_runtime.py, and gmsh_recovery_sidecar.py. | `run_selector_and_recovery` is synthetic/interface-only; it does not supply or implement any of those external components. CLI checks paths and frozen threshold metadata, not checkpoint compatibility. |
| training/ | Reproduce candidate router, shape selector, and optionally feature-backbone training from authorized data. | boundary/stage278p.py, stage279t.py, stage286rp.py, stage286rt.py, stage291p.py. | Stage279T uses Stage275C shards/index; Stage286R-T consumes Stage286R-P train/dev tensors and targets; Stage291P relies on Stage197P/199T/199Training, datasets, p-curve/periodic-UV modules, and a joint-supervision dataset script. |
| adapters/ | Structural interfaces for externally supplied tensor evidence. | `src/breprecover/adapters/protocols.py`; source lineage in Stage197/221/246. | Only a Protocol is present; no concrete licensed/validated feature extractor is bundled. See docs/feature_adapter_contract.md. |

## Runtime Artifact Requirements

The frozen learned route cannot run from source alone. Current research config binds generator checkpoints, a candidate-router checkpoint (Stage279T epoch checkpoint), a shape-selector checkpoint (Stage286R-T best), a Stage236 feature-runtime checkpoint/config, and a Stage294/291 feature-backbone bundle. Paths and hashes are external and are not portable release configuration.

Online inference computes features from the generated candidate graph and runtime model; it is not just a cached-feature lookup. Training/reproduction depends on offline feature shards, labels, manifests, source-group splits, and generated evaluation data. Without those caches and result/complete artifacts, the original research preflights and reported evaluation cannot be reproduced. Synthetic unit tests do not replace that evidence.

## Helpers Requiring Extraction or a Decision

- Current prototype files: `src/breprecover/selection/models.py`, `shape_evidence.py`, `pipeline.py`, `src/breprecover/recovery/decision.py`, and `src/breprecover/cli.py`. Network/input/aggregate tests use optional AST-isolated source parity; the connected selection API uses synthetic tensor weights and contains no CAD feature adapter/backend.

- boundary/stage244r5a.py build_selector_input: directly called by the candidate router; preserve exact feature ordering/normalization or explicitly version a replacement.
- boundary/stage221o.py: supplies online candidate regeneration and hard geometric eligibility.
- inference/stage236_shadow.py and boundary/stage199t.py: supply the feature-model runtime; compatible weights/configuration remain required.
- boundary/stage278p.py and stage286rp.py: architectures must match frozen checkpoints exactly.
- inference/stage288i_mainline.py and stage294x_runtime.py: define fail-closed routing, official-output preservation, and runtime upgrade path. Distill their hash/phase glue; do not silently delete behavior.
- inference/brepBuild.py, primitive_fitting.py, surface_policy.py, trim_reconstruction.py, topology.transfer: current recovery-sidecar dependencies with unresolved source/dependency provenance.
- inference/brep_diagnostics.py and OCC STEP checks: independent acceptance predicates; do not replace with selector confidence.
- Stage295G/306G/316G are evaluation orchestration, not runtime core. Preserve cohort and paired-replay protocols in documentation if runners are omitted.
