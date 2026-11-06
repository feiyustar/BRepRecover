# Dependency and OCCT Validation Audit

## Local probe

The optional adapter was probed in the existing `DTGBrepGen` conda environment;
the environment was not changed. The installed versions reported during the
probe were Python 3.10.13, pythonocc-core 7.5.1, OCCT 7.5.1, Gmsh 4.15.2, and
occwl 3.0.0. The adapter imports only public `OCC.Core` APIs and has no runtime
dependency on Gmsh or occwl. `pyproject.toml` intentionally does not install
pythonocc-core; users must configure the optional CAD runtime themselves.

A programmatically-created box passed kernel validity, one-solid, boundary
edge, STEP write/read, and re-read acceptance checks. A compound containing
two boxes was kernel-valid but rejected by the one-solid predicate. Removing
one box face made `BRepCheck_Analyzer` reject the shape; in the tested OCCT
binding, the research-matched open-wire and bad-shell counters remained zero.
This documents that those counters are specific diagnostics, not a complete
open-boundary or geometric-quality audit. A null `TopoDS_Shape`, missing STEP
path, and non-STEP text file failed closed without an uncaught Python
exception. Tests create only synthetic geometry and temporary files.

## Predicate provenance and implementation

The research semantics were inspected in
`/home/fy/Brep/DTGBrepGen/inference/gmsh_recovery_sidecar.py::_strict` and
`inference/brep_diagnostics.py::analyze_occ_shape`. Those paths use
`BRepCheck_Analyzer(...).IsValid()`, count `TopAbs_SOLID`, count edges on
`ShapeAnalysis_FreeBounds(...).GetOpenWires()`, inspect shells with
`ShapeAnalysis_Shell.HasBadEdges()`, and require those checks again after STEP
round-trip. The independent adapter in
`src/breprecover/validation/occ_validator.py` expresses these predicates with
public pythonocc APIs; it does not copy or call the research sidecar, builder,
or recovery implementation. This extraction is not a clean-room or rights
clearance claim. Provenance/ownership still requires author confirmation.

STEP acceptance checks writer transfer status, writer status, output presence
and nonzero size, reader status, at least one transferred root, a non-null
shape, and the same three shape predicates on the re-read shape. Its temporary
STEP workspace is removed by `TemporaryDirectory` on ordinary Python control
flow. This adapter only validates a supplied shape; it does not heal, mutate,
or construct CAD geometry. It does not provide a disk transaction guarantee.

## Installed license metadata (not redistribution permission)

The local conda package records reported pythonocc-core 7.5.1 as LGPL-3.0 and
OCCT 7.5.1 as LGPL-2.1-only. The installed OCCT tree contains
`share/doc/opencascade/LICENSE_LGPL_21.txt`; the local copy does not establish
whether all relevant additional-exception text is present. Gmsh 4.15.2
metadata/license text reports GPL v2 or later with an exception. The local
occwl 3.0.0 conda record indicates BSD, but its installed package did not
include an obvious license file and upstream repository metadata needs
confirmation. These observations are not a legal interpretation, do not
resolve the exact package/build terms, and do not grant redistribution rights.
No Gmsh or occwl implementation is included here.

Official project references consulted:

- OCCT license overview: <https://www.occt3d.com/dev/doc/overview/html/occt_public_license.html>
- OCCT documentation: <https://www.occt3d.com/dev/doc/overview/html/index.html>
- Gmsh license and project page: <https://gmsh.info/>
- pythonocc-core upstream license: <https://github.com/tpaviot/pythonocc-core/blob/master/LICENSE>
- occwl upstream repository: <https://github.com/AutodeskAILab/occwl>

The online upstream pages may not describe the exact locally installed
historical builds. Before any distribution, verify package archives, notices,
applicable exceptions, dependencies, and rights with the respective rights
holders/advisers.

## Safety boundary and open issues

OCC calls run in-process. Python exceptions are converted to failed checks,
but a malformed native shape may still cause a process crash or indefinite
native call; this adapter cannot enforce a timeout or isolate such failures.
Any production recovery backend that processes untrusted/large CAD should use
the path-based worker with a real wall-clock limit and cleanup policy. The
worker does not implement recovery. The tests establish only behavior for
small synthetic shapes in the probed local runtime, not broad STEP
interoperability, arbitrary inputs, geometry fidelity, or engineering
readiness. Exact pythonocc/OCCT license and redistribution review remains
open.

`IsolatedSTEPValidator` is a separate path-based supervisor/worker protocol.
The caller supplies an existing STEP filename; it does not attempt to
serialize a live OCC object in the parent process. The worker reads the source
STEP, measures its loaded shape, then uses a temporary STEP export and readback
for the fourth predicate. The parent imports no OCC module, captures no
unbounded stdout/stderr, verifies a 64 KiB-bounded JSON result with four
exact-bool fields, and computes aggregate acceptance itself. Timeout kills and
waits for the worker process (POSIX process group where available); it includes
worker startup and OCC work after process creation, but not parent setup or
process-creation latency. `TMPDIR`, `TEMP`, and `TMP` point inside the
supervisor-owned temporary workspace; request, result, and OCC exports are
removed on ordinary completion, timeout, and propagated interruption. This is
not a security sandbox and
cannot guarantee recovery from host-level native/system failures. The
supervisor does not replace the in-memory validator protocol and is not wired
to a recovery coordinator.
