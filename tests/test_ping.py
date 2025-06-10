import pytest
from modules import ping
from modules.ping import ping as ping_function

def test_register_intents():
    intents = ping.register_intents()
    assert isinstance(intents, dict)
    assert "ping" in intents
    assert callable(intents["ping"])
    assert "check connectivity" in intents
    assert callable(intents["check connectivity"])

def test_ping_smoke(monkeypatch):
    monkeypatch.setattr(ping, "speak", lambda msg: None)
    # This is a smoke test; actual ping is not performed
    try:
        ping_function("localhost")
    except Exception:
        pytest.fail("ping() raised an exception on localhost")
