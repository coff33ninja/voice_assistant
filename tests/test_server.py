import pytest
from modules import server

def test_register_intents():
    intents = server.register_intents()
    assert isinstance(intents, dict)
    assert "start server" in intents
    assert callable(intents["start server"])
    assert "stop server" in intents
    assert callable(intents["stop server"])

def test_server_smoke(monkeypatch):
    monkeypatch.setattr(server, "speak", lambda msg: None)
    # This is a smoke test; actual server start/stop is not performed
    try:
        server.start_server()
        server.stop_server()
    except Exception:
        pytest.fail("start_server() or stop_server() raised an exception")
