# Command-Line Usage

## Validate an Existing STEP File

Install the package in an environment that already provides a compatible
`pythonocc-core`/OCCT runtime. The project does not install or pin the CAD
runtime automatically. For example, after activating that environment:

```bash
python -m pip install .
breprecover validate-step /path/to/model.step --timeout 30
```

The command starts an isolated Python worker to read the STEP file and check
OCCT validity, exactly one solid, the configured boundary-edge predicates,
and STEP export/read-back validity. It only validates the input; it does not
repair or rewrite it. A successful result does not establish geometric
fidelity, manufacturing suitability, or broad interoperability.

For a completed STEP check, the command prints one JSON report to standard
output. Argument, input-path, and report-writing errors print a diagnostic to
standard error and do not emit a success-shaped JSON report. Exit codes are:

- `0`: all four acceptance predicates passed.
- `1`: the worker completed its checks and one or more geometric/STEP
  acceptance predicates rejected the file.
- `2`: invalid CLI arguments, missing/non-file input, worker launch/timeout,
  unavailable OCCT runtime, worker exception or malformed response, or
  report-writing failure.

To request a separate report file, the destination must not already exist and
must not refer to the input STEP file:

```bash
breprecover validate-step /path/to/model.step \
  --timeout 30 \
  --python-executable /path/to/env/bin/python \
  --report /path/to/new-validation-report.json
```

The report is atomically created without replacement. If it already exists,
the command fails instead of overwriting it. A failed report write returns
exit code `2` and does not print a success report. The input STEP is never
modified by this command.

`--python-executable` selects the Python environment for the worker. By
default it uses the current interpreter. The worker's imports and OCCT runtime
must be available in that environment. The parent process does not import
OCC; worker stdout/stderr are suppressed and structured diagnostics are
included in the bounded JSON report.

The worker is process-isolated with a wall-clock timeout, but it is not a
security sandbox. It performs no healing, mesh reconstruction, or generator
integration. The predicates are limited to the documented checks and are not
a complete open-boundary or geometric-quality audit.

## Validate a Runtime Configuration

`breprecover validate-config CONFIG.json` checks whether configured external
artifact paths exist and whether the configured shape threshold matches the
threshold recorded in the supplied metadata. It does not load weights, check
their compatibility, or run inference.
