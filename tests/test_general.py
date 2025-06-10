import pytest
from modules import general

def test_register_intents():
    intents = general.register_intents()
    assert isinstance(intents, dict)
    assert "hello" in intents
    assert callable(intents["hello"])
    assert "help" in intents
    assert callable(intents["help"])

def test_hello_smoke(monkeypatch):
    monkeypatch.setattr(general, "speak", lambda msg: None)
    try:
        general.hello()
    except Exception:
        pytest.fail("hello() raised an exception")
