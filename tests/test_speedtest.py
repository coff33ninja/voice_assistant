import pytest
from modules import speedtest

def test_register_intents():
    intents = speedtest.register_intents()
    assert isinstance(intents, dict)
    assert "speed test" in intents
    assert callable(intents["speed test"])
    assert "internet speed" in intents
    assert callable(intents["internet speed"])

def test_speedtest_smoke(monkeypatch):
    monkeypatch.setattr(speedtest, "speak", lambda msg: None)
    # This is a smoke test; actual speed test is not performed
    try:
        speedtest.run_speedtest()
    except Exception:
        pytest.fail("run_speedtest() raised an exception")
