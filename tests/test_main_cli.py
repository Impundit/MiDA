import importlib
import sys
import types
from pathlib import Path

import pytest


@pytest.fixture()
def main_module(monkeypatch):
    """Import the ``main`` module with lightweight stand-ins for heavy deps."""

    project_root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(project_root))

    dummy_read_log = types.ModuleType("read_log")
    dummy_mida = types.ModuleType("MiDA")

    class _ReadLog:
        def __init__(self, eventlog):
            self.eventlog = eventlog

        def readView(self):  # pragma: no cover - placeholder only
            return None

    class _MiDA:
        def __init__(self, eventlog):
            self.eventlog = eventlog

        def optimize(self):  # pragma: no cover - placeholder only
            return None

    dummy_read_log.ReadLog = _ReadLog
    dummy_mida.MiDA = _MiDA

    monkeypatch.setitem(sys.modules, "read_log", dummy_read_log)
    monkeypatch.setitem(sys.modules, "MiDA", dummy_mida)

    sys.modules.pop("main", None)

    return importlib.import_module("main")


def test_main_requires_eventlog(main_module):
    with pytest.raises(SystemExit) as excinfo:
        main_module.main([])
    assert excinfo.value.code == 2


def test_main_invokes_pipeline(main_module, monkeypatch):
    captured = {}

    class DummyReadLog:
        def __init__(self, eventlog):
            captured.setdefault("readlog_init", []).append(eventlog)

        def readView(self):
            captured["readlog_readview"] = True

    class DummyMiDA:
        def __init__(self, eventlog):
            captured.setdefault("mida_init", []).append(eventlog)

        def optimize(self):
            captured["mida_optimize"] = True

    monkeypatch.setattr(main_module, "ReadLog", DummyReadLog)
    monkeypatch.setattr(main_module, "MiDA", DummyMiDA)

    main_module.main(["sample_log"])

    assert captured["readlog_init"] == ["sample_log"]
    assert captured["mida_init"] == ["sample_log"]
    assert captured["readlog_readview"] is True
    assert captured["mida_optimize"] is True
