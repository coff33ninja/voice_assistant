import pytest
import importlib
from modules import find_devices

def test_register_intents():
    intents = find_devices.register_intents()
    assert isinstance(intents, dict)
    assert "find devices" in intents
    assert callable(intents["find devices"])
    assert "scan network" in intents
    assert callable(intents["scan network"])

def test_find_devices_smoke(monkeypatch):
    monkeypatch.setattr(find_devices, "speak", lambda msg: None)
    monkeypatch.setattr(find_devices, "subprocess", importlib.import_module("subprocess"))
    # This is a smoke test; actual network scan is not performed
    try:
        find_devices.find_devices()
    except Exception:
        pytest.fail("find_devices() raised an exception")
