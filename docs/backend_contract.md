# Recovery Backend Contract

This is a prospective interface and acceptance specification. BRepRecover
does not contain a generator integration, Gmsh/OpenCASCADE builder, STEP I/O
backend, or concrete independent validator. `run_selector_and_recovery` only
composes external adapter, selector, backend, and validator interfaces.

## Call Ordering And Original Result

When the official result is valid, the coordinator returns it immediately
without calling the feature adapter, selector, backend, or validator. For an
invalid result, recovery is invoked only when the loaded selector's shape
score meets the same `LoadedSelector.threshold`; no separate recovery
threshold is permitted.

The backend must treat the official result as immutable: work on an isolated
copy/snapshot and do not edit the original object in place. Place intermediate
STEP, mesh, and worker artifacts in a unique temporary workspace; clean it on
success, rejection, exception, timeout, and process termination. A persistent
audit record, if requested, is separate from acceptance. The coordinator
provides no object immutability or filesystem transaction guarantee.

## Timeout And Abstention

The backend must enforce a real wall-clock budget, preferably through a child
process with an OS-level timeout. Passing `timeout_seconds` to a potentially
unbounded in-process call is not enforcement. Timeout, nonzero child exit,
unreadable result, construction/validation exception, or cleanup failure must
abstain and preserve the official result.

Research reference: `inference/gmsh_recovery_sidecar.py` uses
`subprocess.run(..., timeout=...)`, handles `TimeoutExpired`, and scopes
temporary files with `TemporaryDirectory`. This source is not copied or wired
into BRepRecover.

## Independent Acceptance

All four checks are required independently before a recovered candidate may
be returned:

1. **Kernel validity:** OCCT `BRepCheck_Analyzer(shape).IsValid()` is true.
2. **Single solid:** topology analysis counts exactly one `TopAbs_SOLID`.
3. **Boundary edges:** no free edges and no shells with bad edges are reported.
4. **STEP round-trip:** write to STEP, read back, then repeat kernel validity,
   single-solid, and boundary-edge checks on the re-read shape.

These correspond to separate `AcceptanceReport` booleans, informed by
`_strict` in the research sidecar and `analyze_occ_shape` in
`inference/brep_diagnostics.py`. A pre-STEP valid shape alone does not satisfy
the round-trip predicate; all four must be built-in `True`.

Volume/area/mass-like properties or geometric-fidelity metrics, if later
computed, are post-hoc diagnostics only. They cannot substitute for or weaken
these four gates. The cited sidecar acceptance path does not use such
properties as gates.

## Status

No concrete geometry recovery backend is connected. `OCCTShapeValidator`
validates an in-memory shape in the current process; callers must not treat it
as crash- or timeout-isolated. `IsolatedSTEPValidator` is a separate,
file-based API. It accepts an existing STEP path rather than serializing an
arbitrary in-memory OCC object; its worker imports the STEP, checks the shape,
then exports and re-reads a temporary STEP to produce the four predicates.
The parent process handles only paths, subprocess control, and bounded JSON
results; it does not import OCC. On timeout, it terminates the worker process
group on POSIX (or kills the process elsewhere) and waits before cleanup.

The timeout covers waiting for the spawned process after creation, including
worker startup/import, STEP parsing, OCC checks, temporary export, and
round-trip parsing. It does not cover parent-side temporary request creation,
process creation latency before a PID is returned, or cleanup after termination.
The worker has no shell command construction and accepts only a versioned
JSON request; result predicates must all be present and exact built-in bools,
and parent-side acceptance is recomputed from them. Stdout/stderr are discarded
to avoid unbounded capture; diagnostics are returned in a size-limited JSON
file. Temporary workspace is removed after normal completion, rejection,
timeout, and propagated interruptions. Worker `TMPDIR`, `TEMP`, and `TMP` all
point inside this workspace so OCC temporary exports are covered by cleanup.

This is process separation, not a security sandbox. A native/system failure
can still affect the host, and external processes not in the worker process
group are not controlled. No healing, generator integration, construction,
or disk transaction is provided. Synthetic tests establish selected local
runtime behavior only, not broad STEP interoperability, arbitrary CAD safety,
or engineering readiness.
