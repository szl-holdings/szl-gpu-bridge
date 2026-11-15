"""Fail-closed tests for the v3.2 zeus-ml energy window (no zeus/GPU required)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import salad_worker as sw


@pytest.fixture(autouse=True)
def _clean_state():
    sw._ZEUS_STATE.update({"monitor": None, "window": False, "tried": True})
    yield
    sw._ZEUS_STATE.update({"monitor": None, "window": False, "tried": True})


class _FakeMeasurement:
    def __init__(self, total_energy):
        self.total_energy = total_energy


class _FakeMonitor:
    def __init__(self, *args, **kwargs):
        # accepts ZeusMonitor(gpu_indices=[0]) construction kwargs
        self.fail_on_end = kwargs.get("fail_on_end", False)
        self.total_energy = kwargs.get("total_energy", 123.4)
        self.open = []
    def begin_window(self, name):
        self.open.append(name)
    def end_window(self, name):
        if self.fail_on_end:
            raise RuntimeError("boom")
        return _FakeMeasurement(self.total_energy)


def test_begin_unavailable_without_zeus(monkeypatch):
    monkeypatch.setattr(sw, "_ensure_zeus", lambda: None)
    out = sw.zeus_begin_window()
    assert out["label"] == "UNAVAILABLE"
    assert out["joules"] is None
    assert "NOT fabricated" in out["note"]


def test_end_without_window_is_unavailable():
    out = sw.zeus_end_window()
    assert out["label"] == "UNAVAILABLE"
    assert out["joules"] is None


def test_begin_open_failure_is_unavailable(monkeypatch):
    def _boom():
        raise ImportError("zeus missing")
    monkeypatch.setattr(sw, "_ensure_zeus", _boom)
    out = sw.zeus_begin_window()
    assert out["label"] == "UNAVAILABLE"
    assert out["joules"] is None


def test_measured_window_roundtrip(monkeypatch):
    fake = _FakeMonitor()
    monkeypatch.setattr(sw, "_ensure_zeus", lambda: _FakeMonitor)
    beg = sw.zeus_begin_window()
    assert beg["label"] == "MEASURED"
    end = sw.zeus_end_window()
    assert end["label"] == "MEASURED"
    assert end["joules"] == 123.4
    assert end["method"].startswith("zeus-ml")


def test_window_end_failure_is_unavailable(monkeypatch):
    fake = _FakeMonitor(fail_on_end=True)
    monkeypatch.setattr(sw, "_ensure_zeus", lambda: _FakeMonitor)
    sw.zeus_begin_window()
    sw._ZEUS_STATE["monitor"] = fake
    end = sw.zeus_end_window()
    assert end["label"] == "UNAVAILABLE"
    assert end["joules"] is None
    assert "NOT fabricated" in end["note"]


def test_negative_energy_is_unavailable(monkeypatch):
    monkeypatch.setattr(sw, "_ensure_zeus", lambda: _FakeMonitor)
    sw.zeus_begin_window()
    sw._ZEUS_STATE["monitor"] = _FakeMonitor(total_energy=-5.0)
    end = sw.zeus_end_window()
    assert end["label"] == "UNAVAILABLE"
    assert end["joules"] is None
