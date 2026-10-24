import json
import os

import pytest

from stokes_ft.gpulock import gpu_lock
from stokes_ft.provenance import atomic_path, write_json


def test_cpu_work_takes_no_lock(tmp_path):
    with gpu_lock("cpu", path=tmp_path / "gpu.lock") as held:
        assert held is None
    assert not (tmp_path / "gpu.lock").exists()


def test_lock_is_exclusive_and_records_the_holder(tmp_path):
    path = tmp_path / "gpu.lock"
    with gpu_lock("mps", path=path, log=lambda _: None):
        assert f"pid={os.getpid()}" in path.read_text()
        # flock is per open file description, so a second acquisition in the same process also waits
        with pytest.raises(TimeoutError):
            with gpu_lock("mps", path=path, poll_seconds=0.01, timeout=0.05, log=lambda _: None):
                pass
    with gpu_lock("cuda", path=path, timeout=0, log=lambda _: None):
        pass


def test_failed_write_keeps_the_previous_file(tmp_path):
    path = tmp_path / "record.json"
    write_json(path, {"iterations": 10000})
    with pytest.raises(RuntimeError):
        with atomic_path(path) as tmp:
            tmp.write_text("{trunc")
            raise RuntimeError("killed mid-write")
    assert json.loads(path.read_text()) == {"iterations": 10000}
    assert [p.name for p in tmp_path.iterdir()] == ["record.json"]


def test_temporary_name_keeps_the_suffix(tmp_path):
    with atomic_path(tmp_path / "graph_3.vtp") as tmp:
        assert tmp.suffix == ".vtp" and tmp.parent == tmp_path
        tmp.write_text("x")
    assert (tmp_path / "graph_3.vtp").read_text() == "x"
