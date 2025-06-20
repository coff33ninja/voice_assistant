import pytest
from unittest.mock import Mock, patch, MagicMock
import requests
from datetime import datetime, timedelta
import json

from weather_service import WeatherService, WeatherData
from exceptions import WeatherServiceError, InvalidLocationError, APIError

@pytest.fixture
def mock_weather_response():
    return {
        "location": "New York, NY",
        "temperature": 22.5,
        "humidity": 65,
        "description": "Partly cloudy",
        "wind_speed": 5.2,
        "timestamp": "2023-12-01T12:00:00Z"
    }

@pytest.fixture
def invalid_location_response():
    return {"error": "Location not found", "code": 404}

@pytest.fixture
def weather_service():
    with patch('weather_service.requests') as mock_requests:
        service = WeatherService(api_key="test_api_key")
        service._requests = mock_requests
        return service

class TestWeatherServiceInit:
    def test_init_with_valid_api_key(self):
        service = WeatherService(api_key="valid_key")
        assert service.api_key == "valid_key"
        assert service.base_url is not None
        assert service.timeout == 30

    def test_init_with_custom_timeout(self):
        service = WeatherService(api_key="key", timeout=60)
        assert service.timeout == 60

    def test_init_without_api_key_raises_error(self):
        with pytest.raises(ValueError, match="API key is required"):
            WeatherService(api_key=None)

    def test_init_with_empty_api_key_raises_error(self):
        with pytest.raises(ValueError, match="API key cannot be empty"):
            WeatherService(api_key="")

class TestGetCurrentWeather:
    def test_get_weather_by_city_success(self, weather_service, mock_weather_response):
        weather_service._requests.get.return_value.status_code = 200
        weather_service._requests.get.return_value.json.return_value = mock_weather_response

        result = weather_service.get_current_weather("New York, NY")

        assert isinstance(result, WeatherData)
        assert result.location == "New York, NY"
        assert result.temperature == 22.5
        assert result.humidity == 65

    def test_get_weather_by_coordinates_success(self, weather_service, mock_weather_response):
        weather_service._requests.get.return_value.status_code = 200
        weather_service._requests.get.return_value.json.return_value = mock_weather_response

        result = weather_service.get_current_weather(lat=40.7128, lon=-74.0060)

        assert isinstance(result, WeatherData)
        weather_service._requests.get.assert_called_once()
        call_args = weather_service._requests.get.call_args
        assert "lat=40.7128" in str(call_args)

    def test_get_weather_with_units_parameter(self, weather_service, mock_weather_response):
        weather_service._requests.get.return_value.status_code = 200
        weather_service._requests.get.return_value.json.return_value = mock_weather_response

        result = weather_service.get_current_weather("London", units="metric")

        weather_service._requests.get.assert_called_once()
        call_args = weather_service._requests.get.call_args
        assert "units=metric" in str(call_args)

class TestWeatherServiceErrorHandling:
    def test_api_key_unauthorized_error(self, weather_service):
        weather_service._requests.get.return_value.status_code = 401
        weather_service._requests.get.return_value.json.return_value = {"error": "Unauthorized"}

        with pytest.raises(APIError, match="Unauthorized"):
            weather_service.get_current_weather("New York")

    def test_location_not_found_error(self, weather_service):
        weather_service._requests.get.return_value.status_code = 404
        weather_service._requests.get.return_value.json.return_value = {"error": "Location not found"}

        with pytest.raises(InvalidLocationError, match="Location not found"):
            weather_service.get_current_weather("InvalidCity")

    def test_network_timeout_error(self, weather_service):
        weather_service._requests.get.side_effect = requests.exceptions.Timeout()

        with pytest.raises(WeatherServiceError, match="Request timeout"):
            weather_service.get_current_weather("New York")

    def test_connection_error(self, weather_service):
        weather_service._requests.get.side_effect = requests.exceptions.ConnectionError()

        with pytest.raises(WeatherServiceError, match="Connection error"):
            weather_service.get_current_weather("New York")

    def test_invalid_json_response(self, weather_service):
        weather_service._requests.get.return_value.status_code = 200
        weather_service._requests.get.return_value.json.side_effect = json.JSONDecodeError("msg", "doc", 0)

        with pytest.raises(WeatherServiceError, match="Invalid response format"):
            weather_service.get_current_weather("New York")

    def test_missing_required_fields_in_response(self, weather_service):
        incomplete_response = {"location": "New York"}
        weather_service._requests.get.return_value.status_code = 200
        weather_service._requests.get.return_value.json.return_value = incomplete_response

        with pytest.raises(WeatherServiceError, match="Missing required fields"):
            weather_service.get_current_weather("New York")

class TestWeatherServiceEdgeCases:
    def test_empty_location_string(self, weather_service):
        with pytest.raises(ValueError, match="Location cannot be empty"):
            weather_service.get_current_weather("")

    def test_none_location(self, weather_service):
        with pytest.raises(ValueError, match="Location is required"):
            weather_service.get_current_weather(None)

    def test_very_long_location_name(self, weather_service, mock_weather_response):
        long_location = "A" * 1000
        weather_service._requests.get.return_value.status_code = 200
        weather_service._requests.get.return_value.json.return_value = mock_weather_response

        result = weather_service.get_current_weather(long_location)
        assert isinstance(result, WeatherData)

    def test_special_characters_in_location(self, weather_service, mock_weather_response):
        special_location = "São Paulo, Brazil"
        weather_service._requests.get.return_value.status_code = 200
        weather_service._requests.get.return_value.json.return_value = mock_weather_response

        result = weather_service.get_current_weather(special_location)
        assert isinstance(result, WeatherData)

    def test_extreme_coordinates(self, weather_service, mock_weather_response):
        weather_service._requests.get.return_value.status_code = 200
        weather_service._requests.get.return_value.json.return_value = mock_weather_response

        result = weather_service.get_current_weather(lat=90.0, lon=180.0)
        assert isinstance(result, WeatherData)

    def test_invalid_coordinates_range(self, weather_service):
        with pytest.raises(ValueError, match="Invalid latitude"):
            weather_service.get_current_weather(lat=91.0, lon=0.0)

        with pytest.raises(ValueError, match="Invalid longitude"):
            weather_service.get_current_weather(lat=0.0, lon=181.0)

class TestWeatherServiceAdvancedFeatures:
    @patch('weather_service.time.time')
    def test_response_caching(self, mock_time, weather_service, mock_weather_response):
        mock_time.return_value = 1000
        weather_service._requests.get.return_value.status_code = 200
        weather_service._requests.get.return_value.json.return_value = mock_weather_response

        result1 = weather_service.get_current_weather("New York")
        mock_time.return_value = 1030
        result2 = weather_service.get_current_weather("New York")

        assert weather_service._requests.get.call_count == 1
        assert result1.temperature == result2.temperature

    def test_cache_expiration(self, weather_service, mock_weather_response):
        with patch('weather_service.time.time') as mock_time:
            mock_time.return_value = 1000
            weather_service._requests.get.return_value.status_code = 200
            weather_service._requests.get.return_value.json.return_value = mock_weather_response

            weather_service.get_current_weather("New York")

            mock_time.return_value = 2000
            weather_service.get_current_weather("New York")

        assert weather_service._requests.get.call_count == 2

    def test_rate_limiting(self, weather_service):
        weather_service._requests.get.return_value.status_code = 429
        weather_service._requests.get.return_value.json.return_value = {"error": "Rate limit exceeded"}

        with pytest.raises(WeatherServiceError, match="Rate limit exceeded"):
            weather_service.get_current_weather("New York")

class TestWeatherServiceIntegration:
    def test_multiple_location_requests(self, weather_service, mock_weather_response):
        weather_service._requests.get.return_value.status_code = 200
        weather_service._requests.get.return_value.json.return_value = mock_weather_response

        locations = ["New York", "London", "Tokyo", "Sydney"]
        results = []

        for location in locations:
            result = weather_service.get_current_weather(location)
            results.append(result)

        assert len(results) == 4
        assert all(isinstance(r, WeatherData) for r in results)
        assert weather_service._requests.get.call_count == 4

    def test_concurrent_requests_handling(self, weather_service, mock_weather_response):
        import threading

        weather_service._requests.get.return_value.status_code = 200
        weather_service._requests.get.return_value.json.return_value = mock_weather_response

        results = []
        errors = []

        def make_request():
            try:
                result = weather_service.get_current_weather("Test City")
                results.append(result)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=make_request) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 5

    def test_api_response_time_logging(self, weather_service, mock_weather_response):
        with patch('weather_service.logger') as mock_logger:
            weather_service._requests.get.return_value.status_code = 200
            weather_service._requests.get.return_value.json.return_value = mock_weather_response

            weather_service.get_current_weather("New York")

            mock_logger.info.assert_called()
            log_message = mock_logger.info.call_args[0][0]
            assert "response_time" in log_message.lower()
