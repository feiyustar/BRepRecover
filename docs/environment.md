# Environment

## CPU Prototype and Tests

Use Python 3.10 for this recorded-version recipe. The package's broader Python
metadata is not a claim that these historical wheels support every newer
Python release. For a CPU-only PyTorch install, select the CPU wheel
index for your platform from the [official PyTorch install selector](https://pytorch.org/get-started/locally/), then install the project and test extras:

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.2.2
python -m pip install -r requirements-core-test.txt
python -m pip install -e '.[test]'
python -m pytest -q
python examples/synthetic_selector_smoke.py
```

The local test runtime used for the reported checks had Python 3.10, torch
2.2.2, numpy 1.26.4, and pytest 9.1.1. These are observations, not a fresh
environment validation or a complete compatibility matrix. Optional exact
test pins are listed in `requirements-core-test.txt` and installed before the
editable package to avoid silently resolving a different NumPy major version.

## Optional CAD Runtime

The optional `pythonocc-core` validator was tested only in an existing
pythonocc-core/OCCT 7.5.1 runtime. The `envs/optional-cad.yml` file records a
candidate conda-forge environment recipe; it has not been validated as a clean
environment creation recipe. Do not interpret its presence as a guarantee of
installation, platform compatibility, or redistribution rights. The CLI
validates existing STEP files only; no healing backend is included.

## Data and Artifacts

No data, checkpoints, feature caches, or results are included. ABC is hosted at
<https://archive.nyu.edu/handle/2451/43778>; DeepCAD project and data
instructions are at <https://github.com/ChrisWu1997/DeepCAD>. Review the
applicable host and dataset terms before downloading or using them. Keep any
locally acquired data, generated features, weights, and results outside this
source tree and under ignored paths such as `data/`, `datasets/`, `features/`,
`weights/`, and `results/`. No preprocessing or training entry point is
provided here.
