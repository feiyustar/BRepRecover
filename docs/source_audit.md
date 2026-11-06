# Source and License Audit

## Scope

This is a targeted review of the local research tree, local DTGBrepGen-base comparison tree, and linked public upstream repository. It is not a complete provenance review, legal opinion, or permission grant. DTGBrepGen and DTGBrepGen-base were read-only.

## License Findings

### Historical findings recorded 2026-11-01

- Neither local tree contains a root LICENSE, COPYING, or NOTICE file.
- Public upstream: https://github.com/jinli99/DTGBrepGen . Its current file listing does not show a license, and the conventional LICENSE path returned 404 during this audit: https://github.com/jinli99/DTGBrepGen/blob/master/LICENSE .
- The upstream README identifies the project and links to its paper and pretrained models, but does not itself provide an explicit source-code license: https://github.com/jinli99/DTGBrepGen/blob/master/README.md .
- Public visibility is not treated as authorization to copy, modify, or redistribute source. At the time of this audit, no LICENSE was created because applicable rights had not been established.
- DeepCAD's public code repository advertises an MIT license: https://github.com/ChrisWu1997/DeepCAD . That applies to that repository's code, not automatically to its data, model weights, or BRepRecover files.
- Third-party runtime/development components, including Gmsh, OpenCASCADE/PythonOCC, occwl, PyTorch, and other dependencies, need release-specific license and notice review. This repository does not vendor them.

### Progress recorded 2026-11-06

- The BRepRecover code-only prototype was made public at https://github.com/feiyustar/BRepRecover under the selected MIT license, with maintainer approval on 2026-11-06. This records repository publication, not an independent legal or provenance audit. Included extracted-behavior provenance and redistribution-rights review remains pending; excluded baseline-derived files, third-party dependencies, data, derived data, and weights are not covered by this publication or license selection.
- MIT was selected for authorized BRepRecover project code, and the root `LICENSE` records `Copyright (c) 2026 BRepRecover contributors`. `pyproject.toml` points package metadata to that file.
- This selection does not assert that upstream DTGBrepGen, baseline-derived files, datasets, derived data, model weights, or dependencies are covered by this license. `inference/generate.py` and `inference/brepBuild.py` remain excluded from the portable prototype and this selection does not clear their provenance or redistribution rights.
- Provenance and rights review for the included extracted behaviors remains pending. The MIT file does not make the current candidate publicly release-ready or establish full-paper reproduction rights.

## File Lineage Review

| Research path reviewed | Baseline comparison | Audit classification | Release handling |
|---|---|---|---|
| inference/generate.py | Same path exists in baseline; differs substantially. Diff adds research orchestration and hooks, including fitted replay. | Derived from baseline plus research additions; authorship and upstream reuse rights are not inferred from the diff. | Do not copy wholesale. Extract only authorized minimal interfaces after permission and provenance review. |
| inference/brepBuild.py | Same path exists in baseline; diff adds diagnostics and changes surface-fit policy/worker behavior. | Derived from baseline and modified research code. | Do not copy wholesale. Builder/healing logic is blocked pending source permission and third-party dependency review. |
| inference/trim_reconstruction.py, inference/brep_diagnostics.py | No same-path baseline counterpart. | Research-tree additions; local origin is plausible, but no file-level provenance record was found. | Author confirmation and review for copied/adapted third-party content required before redistribution. |
| boundary/stage278p.py, stage279t.py, stage286rp.py, stage286rt.py | No same-path baseline counterpart; research modules are untracked in the source tree. | Research selector/model/training implementations, not present in baseline. This does not prove sole authorship or redistribution rights. | Do not migrate stage glue wholesale. Preserve reviewed mathematical contracts after author confirmation. |
| boundary/stage197p.py, stage199t.py, stage199_training.py, stage291p.py | No same-path baseline counterpart for reviewed research modules. | Research model/training path with substantial dependencies on other research and topology modules. | Keep outside the first portable extraction; require provenance, checkpoint, and data review. |
| boundary/stage221o.py, inference/stage246i_mainline.py, stage288i_mainline.py, stage294x_runtime.py | No same-path baseline counterpart; research-tree additions. | Research online-candidate, selector, and runtime adapters. | The current prototype extracts candidate/shape model and aggregate contracts but omits online feature runtime and adapters; extraction is not a clean-room or permission claim. |
| inference/gmsh_recovery_sidecar.py | No same-path baseline counterpart. | Research sidecar; depends on the modified baseline-derived B-rep builder and CAD libraries. | Do not copy until builder lineage and dependency terms are approved. |
| scripts/stage0_infer.py and stage295g/306g/316g entries/configs | Research/evaluation orchestration around generation and frozen artifacts. | Experiment/pipeline adapters, not independent product entry points. Configs contain local paths and hash-bound research state. | Do not port configs or stage guards as release API. Replace only after a clean input/output contract is designed. |
| src/breprecover/selection/shape_evidence.py | Newly written here from an audited feature contract; no source file was copied into this tree. | Behavior extracted from current research implementation; author/ownership confirmation pending. Not described as clean-room. | Prototype only; retain audit link and parity test until owner approval. |
| src/breprecover/selection/models.py | Architecture/input contracts match boundary/stage278p.py, boundary/stage244r5a.py, and boundary/stage286rp.py; no original source files copied. | Behavior extracted into a separately organized prototype; ownership/provenance and redistribution rights remain unconfirmed. No baseline absence is treated as authorization. | Keep exact AST/state-dict parity tests; require author confirmation before release. |
| src/breprecover/recovery/decision.py | New thin control-flow contract informed by Stage288I/Stage294X behavior; no builder or validator source copied. | New orchestration code; upstream behavior inspiration is documented; not a clean-room claim. | Backend and validator are injected; only contract tests exist, not CAD-kernel execution. |
| src/breprecover/selection/pipeline.py | New versioned input adapter and orchestration combining the local extracted networks/aggregator; no research source file copied. | Behavior composes research-derived contracts; authored here, but not a clean-room or rights determination. | Synthetic tensor-only integration tests; external feature semantics and checkpoint provenance remain unauthenticated. |
| src/breprecover/adapters/protocols.py and run_selector_and_recovery | New structural protocol/composition function; no feature extractor, generator, or geometry builder copied. | Interface behavior is informed by Stage288I/Stage294X fail-closed routing; this is not a clean-room or rights determination. | No concrete adapter/backend/validator is included; docs document source semantics and unimplemented obligations. |

## Baseline Diff Notes

inference/brepBuild.py is not byte-identical to baseline: its diff adds callback diagnostics, changes worker configuration, and changes the SurfaceFitPolicy path and surface-fit policy values. It cannot be described as an unchanged upstream builder.

inference/generate.py has extensive research additions and imports the B-rep builder and topology code. It is a mixed upstream-derived/research orchestration file, not a safe copy source.

Most reviewed selector/runtime/training files do not exist at the same path in baseline. That establishes only absence of an exact baseline counterpart; it does not establish independent authorship, absence of other source influences, or redistribution rights.

## Open Questions

1. Obtain written confirmation of ownership and applicable license for baseline-derived code.
2. Obtain author/provenance confirmation for added research modules and document borrowed snippets or third-party adaptations.
3. Review dependency licenses and notices against exact release versions.
4. Determine whether model weights, feature caches, and dataset-derived labels may be redistributed; default assumption here is that they are not included.
