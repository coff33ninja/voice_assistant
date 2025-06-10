import pytest
from modules import wol

def test_register_intents():
    intents = wol.register_intents()
    assert isinstance(intents, dict)
    assert "wake on lan" in intents
    assert callable(intents["wake on lan"])
    assert "wake device" in intents
    assert callable(intents["wake device"])

def test_wake_on_lan_smoke(monkeypatch):
    monkeypatch.setattr(wol, "speak", lambda msg: None)
    # This is a smoke test; actual WOL is not performed
    try:
        wol.wake_on_lan("00:11:22:33:44:55")
    except Exception:
        pytest.fail("wake_on_lan() raised an exception")
