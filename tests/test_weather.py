import pytest
from modules import weather

def test_get_weather_wmo_description():
    assert weather.get_weather_wmo_description(0) == "Clear sky"
    assert weather.get_weather_wmo_description(99) == "Thunderstorm with heavy hail"
    assert weather.get_weather_wmo_description(999) == "an unknown weather condition"

def test_register_intents():
    intents = weather.register_intents()
    assert isinstance(intents, dict)
    assert "get weather" in intents
    assert callable(intents["get weather"])
    assert "weather in" in intents
    assert callable(intents["weather in"])

def test_get_weather_smoke(monkeypatch):
    monkeypatch.setattr(weather, "speak", lambda msg: None)
    # This is a smoke test; actual API call is not performed
    try:
        weather.get_weather("")  # Should handle empty input gracefully
    except Exception:
        pytest.fail("get_weather() raised an exception on empty input")
