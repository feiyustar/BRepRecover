"""Supervisor protocol tests with deterministic fake Python workers."""
from __future__ import annotations

import json
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap

import pytest

from breprecover.validation.isolated_step import _run_worker
import breprecover.validation.isolated_step as isolated_step


def _fake_worker(tmp_path: Path, monkeypatch, name: str, source: str) -> str:
    (tmp_path / f"{name}.py").write_text(textwrap.dedent(source), encoding="utf-8")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    return name


def _outcome(tmp_path: Path, monkeypatch, name: str, source: str, timeout: float = 5):
    module = _fake_worker(tmp_path, monkeypatch, name, source)
    return _run_worker(
        tmp_path / "unused.step",
        timeout_seconds=timeout,
        python_executable=sys.executable,
        worker_module=module,
    )


def test_supervisor_recomputes_acceptance_from_strict_boolean_predicates(tmp_path, monkeypatch):
    outcome = _outcome(
        tmp_path,
        monkeypatch,
        "fake_valid_worker",
        """
        import json, sys
        request = json.load(open(sys.argv[-1], encoding="utf-8"))
        result = {
            "schema": "breprecover.occt-validation-result.v1",
            "acceptance": {
                "kernel_valid": True,
                "single_solid": True,
                "boundary_edges_valid": True,
                "step_roundtrip_valid": True,
            },
            "details": {"synthetic": True},
        }
        with open(request["result_json"], "w", encoding="utf-8") as stream:
            json.dump(result, stream)
        """,
    )
    assert outcome.acceptance.accepted is True
    assert outcome.details["worker"] == {"synthetic": True}


def test_relative_input_path_is_expanded_and_made_absolute(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _fake_worker(
        tmp_path,
        monkeypatch,
        "fake_relative_worker",
        """
        import json, sys
        r=json.load(open(sys.argv[-1]))
        json.dump({"schema":"breprecover.occt-validation-result.v1",
          "acceptance":{"kernel_valid":False,"single_solid":False,
          "boundary_edges_valid":False,"step_roundtrip_valid":False},
          "details":{"path":r["input_step"]}},open(r["result_json"],"w"))
        """,
    )
    outcome = _run_worker(
        "relative.step",
        timeout_seconds=5,
        python_executable=sys.executable,
        worker_module="fake_relative_worker",
    )
    assert outcome.details["worker"]["path"] == str(tmp_path / "relative.step")


@pytest.mark.parametrize(
    "name,source,expected",
    [
        (
            "fake_nonzero_worker",
            "import sys\nsys.exit(7)\n",
            "worker_nonzero_exit",
        ),
        (
            "fake_malformed_worker",
            "import json,sys\nr=json.load(open(sys.argv[-1]));open(r['result_json'],'w').write('{')\n",
            "worker_result_invalid",
        ),
        (
            "fake_missing_worker",
            "pass\n",
            "worker_result_missing",
        ),
        (
            "fake_incomplete_worker",
            """
            import json,sys
            r=json.load(open(sys.argv[-1]))
            json.dump({"schema":"breprecover.occt-validation-result.v1",
                      "acceptance":{"kernel_valid":True},"details":{}},
                      open(r["result_json"],"w"))
            """,
            "worker_result_invalid",
        ),
        (
            "fake_nonbool_worker",
            """
            import json,sys
            r=json.load(open(sys.argv[-1]))
            json.dump({"schema":"breprecover.occt-validation-result.v1",
                      "acceptance":{"kernel_valid":1,"single_solid":True,
                      "boundary_edges_valid":True,"step_roundtrip_valid":True},
                      "details":{}}, open(r["result_json"],"w"))
            """,
            "worker_result_invalid",
        ),
    ],
)
def test_supervisor_rejects_worker_failures(tmp_path, monkeypatch, name, source, expected):
    outcome = _outcome(tmp_path, monkeypatch, name, source)
    assert outcome.acceptance.accepted is False
    assert outcome.details["failure"] == expected


def test_supervisor_timeout_kills_and_waits_for_worker(tmp_path, monkeypatch):
    outcome = _outcome(
        tmp_path,
        monkeypatch,
        "fake_hung_worker",
        "import time\nwhile True: time.sleep(10)\n",
        timeout=1.0,
    )
    assert outcome.acceptance.accepted is False
    assert outcome.details["failure"] == "worker_timeout"
    assert list(tmp_path.glob("breprecover-step-worker-*")) == []


def test_timeout_cleans_worker_temp_location(tmp_path, monkeypatch):
    record_path = tmp_path / "worker-temp-path.txt"
    monkeypatch.setenv("TEST_TMP_RECORD", str(record_path))
    _outcome(
        tmp_path,
        monkeypatch,
        "fake_tempdir_hung_worker",
        """
        import os, tempfile, time
        from pathlib import Path
        Path(os.environ["TEST_TMP_RECORD"]).write_text(tempfile.gettempdir())
        tempfile.mkdtemp(prefix="left-behind-if-not-cleaned-")
        while True: time.sleep(10)
        """,
        timeout=1.0,
    )
    worker_tmp = Path(record_path.read_text(encoding="utf-8"))
    assert worker_tmp.name == "worker-tmp"
    assert not worker_tmp.exists()


def test_keyboard_interrupt_kills_and_waits_for_live_process(tmp_path, monkeypatch):
    class FakeProcess:
        pid = 424242
        returncode = None
        wait_calls = 0

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            self.wait_calls += 1
            if self.wait_calls == 1:
                raise KeyboardInterrupt
            self.returncode = -9
            return self.returncode

    process = FakeProcess()
    monkeypatch.setattr(isolated_step.subprocess, "Popen", lambda *a, **kw: process)
    killed_signals = []

    def fake_killpg(pid, sig):
        killed_signals.append(sig)
        process.returncode = -sig
        raise ProcessLookupError

    monkeypatch.setattr(isolated_step.os, "killpg", fake_killpg)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    with pytest.raises(KeyboardInterrupt):
        _run_worker(
            "relative.step",
            timeout_seconds=10,
            python_executable=sys.executable,
            worker_module="unused_worker",
        )
    assert process.wait_calls >= 2
    assert killed_signals
    assert list(tmp_path.glob("breprecover-step-worker-*")) == []


def test_result_reader_caps_actual_read_size(tmp_path, monkeypatch):
    class BoundedReader(io.BytesIO):
        requested = None

        def read(self, size=-1):
            self.requested = size
            return super().read(size)

    reader = BoundedReader(b"x" * 1000)
    monkeypatch.setattr(Path, "open", lambda self, mode="r", *a, **kw: reader)
    monkeypatch.setattr(isolated_step, "_MAX_RESULT_BYTES", 32)
    with pytest.raises(ValueError, match="worker result size"):
        isolated_step._read_worker_result(Path("fake-result.json"))
    assert reader.requested == 33


def test_parent_validation_import_does_not_load_occ(tmp_path):
    source_root = Path(__file__).resolve().parents[1] / "src"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(source_root)
    script = (
        "import sys; import breprecover.validation.isolated_step; "
        "assert not any(name == 'OCC' or name.startswith('OCC.') for name in sys.modules)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=20,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr.decode("utf-8", errors="replace")
