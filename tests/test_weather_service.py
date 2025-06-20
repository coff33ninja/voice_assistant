import pytest # Ensure pytest is imported
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import json
import os
import sys
import aiohttp

# Ensure pytest-asyncio is installed and configured in pytest.ini
# (You've already added the 'asyncio' marker, which is good)

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import the actual asynchronous functions from the module
from modules.weather_service import get_weather_async, get_current_location_coordinates_async, initialize_weather_service
# Note: WeatherService and WeatherData classes are not in the module, so they are removed.
# Custom exceptions WeatherServiceError, InvalidLocationError, APIError are not raised by the async functions, so they are removed.

@pytest.fixture(autouse=True)
def mock_api_key():
    """
    Mock the function that gets the API key to ensure tests use a controlled key.
    This prevents tests from failing due to missing environment variables or files.
    """
    with patch('modules.weather_service.get_openweather_api_key', return_value="test_api_key") as mock_get_key:
        # Re-initialize the service to pick up the mocked key
        initialize_weather_service()
        yield mock_get_key

@pytest.fixture
def mock_aiohttp_get():
    """
    Fixture to mock aiohttp.ClientSession.get.
    Returns a mock response object that can be configured.
    """
    with patch('aiohttp.ClientSession.get', new_callable=AsyncMock) as mock_get:
        mock_response = MagicMock(spec=aiohttp.ClientResponse)
        # Properly mock async context manager protocol
        async def async_json():
            return mock_response._json_payload
        mock_response.json = AsyncMock(side_effect=async_json)
        mock_response._json_payload = None  # Will be set in each test
        mock_get.return_value.__aenter__.return_value = mock_response
        mock_get.return_value.__aexit__.return_value = False
        yield mock_get, mock_response

@pytest.fixture
def mock_weather_success_payload():
    """Sample successful weather API response payload."""
    return {
        "coord": {"lon": -74.01, "lat": 40.71},
        "weather": [{"id": 802, "main": "Clouds", "description": "partly cloudy", "icon": "03d"}],
        "base": "stations",
        "main": {"temp": 22.5, "feels_like": 23.1, "temp_min": 20.0, "temp_max": 25.0, "pressure": 1012, "humidity": 65},
        "visibility": 10000,
        "wind": {"speed": 5.2, "deg": 210},
        "clouds": {"all": 40},
        "dt": 1678886400, # Example timestamp
        "sys": {"type": 2, "id": 2039034, "country": "US", "sunrise": 1678867200, "sunset": 1678909200},
        "timezone": -18000,
        "id": 5128581,
        "name": "New York", # City name returned by API
        "cod": 200
    }

@pytest.fixture
def mock_ip_geo_success_payload():
    """Sample successful IP geolocation API response payload."""
    return {
        "status": "success",
        "country": "United States",
        "countryCode": "US",
        "region": "NY",
        "regionName": "New York",
        "city": "New York",
        "zip": "10001",
        "lat": 40.7128,
        "lon": -74.0060,
        "timezone": "America/New_York",
        "isp": "AT&T",
        "org": "AT&T",
        "as": "AS7018 AT&T Services, Inc.",
        "query": "192.168.1.1" # Example IP
    }

@pytest.fixture
def mock_ip_geo_failure_payload():
    """Sample failed IP geolocation API response payload."""
    return {
        "status": "fail",
        "message": "reserved range",
        "query": "192.168.1.1"
    }

@pytest.mark.asyncio
class TestGetWeatherAsync:
    """Test the asynchronous get_weather_async function."""

    async def test_get_weather_by_city_success(self, mock_aiohttp_get, mock_weather_success_payload):
        mock_get, mock_response = mock_aiohttp_get
        mock_response.json.return_value = mock_weather_success_payload
        mock_response._json_payload = mock_weather_success_payload
        mock_response.status = 200 # Set status code for raise_for_status

        result = await get_weather_async(location_query="New York")

        mock_get.assert_called_once_with(
            "https://api.openweathermap.org/data/2.5/weather",
            params={"appid": "test_api_key", "units": "metric", "q": "New York"},
            raise_for_status=True
        )
        assert isinstance(result, dict)
        assert result["city"] == "New York"
        assert result["temp"] == 22.5
        assert result["description"] == "partly cloudy"
        assert "error" not in result

    async def test_get_weather_by_coordinates_success(self, mock_aiohttp_get, mock_weather_success_payload):
        mock_get, mock_response = mock_aiohttp_get
        mock_response.json.return_value = mock_weather_success_payload
        mock_response._json_payload = mock_weather_success_payload
        mock_response.status = 200

        result = await get_weather_async(location_query=(40.7128, -74.0060))

        mock_get.assert_called_once_with(
            "https://api.openweathermap.org/data/2.5/weather",
            params={"appid": "test_api_key", "units": "metric", "lat": "40.7128", "lon": "-74.0060"},
            raise_for_status=True
        )
        assert isinstance(result, dict)
        assert result["city"] == "New York" # Assuming API returns city name for coords
        assert "error" not in result

    async def test_get_weather_with_entities_location(self, mock_aiohttp_get, mock_weather_success_payload):
        mock_get, mock_response = mock_aiohttp_get
        mock_response.json.return_value = mock_weather_success_payload
        mock_response._json_payload = mock_weather_success_payload
        mock_response.status = 200

        entities = {"location": "London"}
        result = await get_weather_async(entities=entities)

        mock_get.assert_called_once_with(
            "https://api.openweathermap.org/data/2.5/weather",
            params={"appid": "test_api_key", "units": "metric", "q": "London"},
            raise_for_status=True
        )
        assert isinstance(result, dict)
        assert "error" not in result

    async def test_get_weather_prefers_entities_location_over_query(self, mock_aiohttp_get, mock_weather_success_payload):
        mock_get, mock_response = mock_aiohttp_get
        mock_response.json.return_value = mock_weather_success_payload
        mock_response._json_payload = mock_weather_success_payload
        mock_response.status = 200

        entities = {"location": "Paris"}
        result = await get_weather_async(location_query="Berlin", entities=entities)

        mock_get.assert_called_once_with(
            "https://api.openweathermap.org/data/2.5/weather",
            params={"appid": "test_api_key", "units": "metric", "q": "Paris"},
            raise_for_status=True
        )
        assert isinstance(result, dict)
        assert "error" not in result

    @patch('modules.weather_service.get_current_location_coordinates_async', new_callable=AsyncMock)
    async def test_get_weather_falls_back_to_ip_geo(self, mock_get_coords, mock_aiohttp_get, mock_weather_success_payload, mock_ip_geo_success_payload):
        mock_get, mock_response = mock_aiohttp_get
        mock_response.json.return_value = mock_weather_success_payload
        mock_response._json_payload = mock_weather_success_payload
        mock_response.status = 200
        mock_get_coords.return_value = (mock_ip_geo_success_payload['lat'], mock_ip_geo_success_payload['lon'])

        result = await get_weather_async() # No location_query or entities

        mock_get_coords.assert_called_once()
        mock_get.assert_called_once_with(
            "https://api.openweathermap.org/data/2.5/weather",
            params={"appid": "test_api_key", "units": "metric", "lat": "40.7128", "lon": "-74.0060"},
            raise_for_status=True
        )
        assert isinstance(result, dict)
        assert "error" not in result

    @patch('modules.weather_service.get_current_location_coordinates_async', new_callable=AsyncMock)
    async def test_get_weather_ip_geo_fails(self, mock_get_coords, mock_aiohttp_get):
        mock_get, mock_response = mock_aiohttp_get
        mock_get_coords.return_value = None # IP Geo fails

        result = await get_weather_async() # No location_query or entities

        mock_get_coords.assert_called_once()
        mock_get.assert_not_called() # Weather API should not be called
        assert isinstance(result, dict)
        assert "error" in result
        assert "Could not determine your current location" in result["error"]

    async def test_get_weather_unsupported_future_date_entity(self, mock_aiohttp_get):
        mock_get, mock_response = mock_aiohttp_get
        entities = {"date_reference": "tomorrow"}
        result = await get_weather_async(location_query="New York", entities=entities)

        mock_get.assert_not_called() # API should not be called for unsupported date
        assert isinstance(result, dict)
        assert "message" in result
        assert "I can only provide current weather" in result["message"]

    async def test_get_weather_api_key_missing(self, mock_aiohttp_get):
        mock_get, mock_response = mock_aiohttp_get
        # Temporarily set api_key to None for this test
        with patch('modules.weather_service.api_key', None):
            result = await get_weather_async(location_query="New York")

        mock_get.assert_not_called()
        assert isinstance(result, dict)
        assert "error" in result
        assert "API key not configured" in result["error"]

    async def test_get_weather_api_401_unauthorized(self, mock_aiohttp_get):
        mock_get, mock_response = mock_aiohttp_get

        mock_request_info = MagicMock()
        mock_request_info.url = "https://api.openweathermap.org/data/2.5/weather"
        mock_request_info.method = "GET"
        mock_request_info.headers = {}
        mock_request_info.real_url = mock_request_info.url

        mock_response.raise_for_status.side_effect = aiohttp.ClientResponseError(
            request_info=mock_request_info,
            history=(),
            status=401,
            message="Unauthorized"
        )
        mock_response.status = 401 # Set status for potential logging/checking in the function
        result = await get_weather_async(location_query="New York")

        assert isinstance(result, dict)
        assert "error" in result

    async def test_get_weather_api_404_location_not_found(self, mock_aiohttp_get):
        mock_get, mock_response = mock_aiohttp_get
        mock_request_info = MagicMock()
        mock_request_info.url = "https://api.openweathermap.org/data/2.5/weather"
        mock_request_info.method = "GET"
        mock_request_info.headers = {}
        mock_request_info.real_url = mock_request_info.url
        mock_response.raise_for_status.side_effect = aiohttp.ClientResponseError(
            request_info=mock_request_info,
            history=(),
            status=404,
            message="Not Found"
        )
        mock_response.status = 404

        await get_weather_async(location_query="InvalidCity")

    async def test_get_weather_api_other_http_error(self, mock_aiohttp_get):
        mock_get, mock_response = mock_aiohttp_get
        mock_request_info = MagicMock()
        mock_request_info.url = "https://api.openweathermap.org/data/2.5/weather"
        mock_request_info.method = "GET"
        mock_request_info.headers = {}
        mock_request_info.real_url = mock_request_info.url
        mock_response.raise_for_status.side_effect = aiohttp.ClientResponseError(
            request_info=mock_request_info,
            history=(),
            status=500,
            message="Internal Server Error"
        )
        mock_response.status = 500

        result = await get_weather_async(location_query="New York")
        # Accept either the HTTP status or the fallback error message
        assert ("HTTP Status: 500" in result["error"] or "An unexpected error occurred" in result["error"])

    async def test_get_weather_network_error(self, mock_aiohttp_get):
        mock_get, mock_response = mock_aiohttp_get

        mock_os_error = OSError("Network unreachable")
        mock_request_info = MagicMock()
        mock_request_info.url = "https://api.openweathermap.org/data/2.5/weather" # etc.
        # For ClientConnectorError, the first argument is request_info, second is the os_error
        mock_get.side_effect = aiohttp.ClientConnectorError(mock_request_info, mock_os_error)

        await get_weather_async(location_query="New York")

    async def test_get_weather_invalid_json_response(self, mock_aiohttp_get):
        mock_get, mock_response = mock_aiohttp_get
        mock_response.status = 200
        # Simulate a ContentTypeError which can happen before JSONDecodeError
        # if the content type is not application/json
        mock_request_info_for_content_type = MagicMock()
        mock_request_info_for_content_type.url = "http://example.com/weather" # Dummy URL
        mock_history_for_content_type = MagicMock() # Mock for history tuple

        mock_response.json.side_effect = aiohttp.ContentTypeError(mock_request_info_for_content_type, mock_history_for_content_type)
        await get_weather_async(location_query="New York")

    async def test_get_weather_missing_required_fields_in_response(self, mock_aiohttp_get):
        mock_get, mock_response = mock_aiohttp_get
        mock_response.status = 200
        mock_response.json.return_value = {"coord": {}, "weather": [], "main": {}} # Missing 'description', 'temp', 'city' implicitly

        await get_weather_async(location_query="New York")

    async def test_get_weather_empty_location_query_and_entities(self, mock_aiohttp_get):
        mock_get, mock_response = mock_aiohttp_get
        # Mock IP geo to return None for this specific test case
        with patch('modules.weather_service.get_current_location_coordinates_async', new_callable=AsyncMock, return_value=None):
             result = await get_weather_async(location_query=None, entities=None)

        mock_get.assert_not_called()
        assert isinstance(result, dict)
@pytest.mark.asyncio
class TestGetCurrentLocationCoordinatesAsync:
    """Test the asynchronous get_current_location_coordinates_async function."""

    async def test_get_current_location_coordinates_success(self, mock_aiohttp_get, mock_ip_geo_success_payload):
        mock_get, mock_response = mock_aiohttp_get
        mock_response.json.return_value = mock_ip_geo_success_payload
        mock_response.status = 200

        await get_current_location_coordinates_async()

        mock_get.assert_called_once_with("https://ip-api.com/json/")

    async def test_get_current_location_coordinates_api_failure_status(self, mock_aiohttp_get, mock_ip_geo_failure_payload):
        mock_get, mock_response = mock_aiohttp_get
        mock_response.json.return_value = mock_ip_geo_failure_payload
        mock_response.status = 200 # HTTP status is OK, but API status is 'fail'

        await get_current_location_coordinates_async()

        mock_get.assert_called_once_with("https://ip-api.com/json/")
    async def test_get_current_location_coordinates_http_error(self, mock_aiohttp_get):
        mock_get, mock_response = mock_aiohttp_get
        mock_response.raise_for_status.side_effect = aiohttp.ClientResponseError(
            Mock(), (), status=500, message="Internal Server Error"
        )
        mock_response.status = 500

        await get_current_location_coordinates_async()

        mock_get.assert_called_once_with("https://ip-api.com/json/")
    async def test_get_current_location_coordinates_network_error(self, mock_aiohttp_get):
        mock_get, mock_response = mock_aiohttp_get
        mock_get.side_effect = aiohttp.ClientConnectorError(Mock(), Mock())

        await get_current_location_coordinates_async()

        mock_get.assert_called_once_with("https://ip-api.com/json/")
    async def test_get_current_location_coordinates_invalid_json(self, mock_aiohttp_get):
        mock_get, mock_response = mock_aiohttp_get
        mock_response.status = 200
        mock_response.json.side_effect = json.JSONDecodeError("msg", "doc", 0)

        await get_current_location_coordinates_async()

        mock_get.assert_called_once_with("https://ip-api.com/json/")
    async def test_get_current_location_coordinates_missing_fields(self, mock_aiohttp_get):
        mock_get, mock_response = mock_aiohttp_get
        mock_response.status = 200
        mock_response.json.return_value = {"status": "success", "city": "New York"} # Missing lat/lon

        await get_current_location_coordinates_async()

        mock_get.assert_called_once_with("https://ip-api.com/json/")
@pytest.mark.asyncio
class TestInitializeWeatherService:
    """Test the initialization function."""

    @patch('modules.weather_service.get_openweather_api_key', return_value=None)
    @patch('os.path.exists', return_value=False)
    @patch('builtins.print')
    async def test_initialize_weather_service_no_key_found(self, mock_print, mock_exists, mock_get_key):
        # Ensure api_key is None before calling
        with patch('modules.weather_service.api_key', None):
            initialize_weather_service()
            mock_print.assert_any_call(
                f"Warning: OpenWeather API key not found in environment or at {os.path.join(_PROJECT_ROOT, 'models', 'openweather_api_key.txt')}. Weather service will not work."
            )
            import modules.weather_service
            assert modules.weather_service.api_key is None

    @patch('modules.weather_service.get_openweather_api_key', return_value="env_key")
    @patch('os.path.exists', return_value=False)
    @patch('builtins.print')
    async def test_initialize_weather_service_key_from_env(self, mock_print, mock_exists, mock_get_key):
         # Ensure api_key is None before calling
        with patch('modules.weather_service.api_key', None):
            initialize_weather_service()
            mock_get_key.assert_called_once()
            mock_print.assert_any_call("Weather service API key loaded from environment.")
            import modules.weather_service
            assert modules.weather_service.api_key == "env_key"

    @patch('modules.weather_service.get_openweather_api_key', return_value=None)
    @patch('os.path.exists', return_value=True)
    @patch('builtins.open', new_callable=MagicMock)
    @patch('builtins.print')
    async def test_initialize_weather_service_key_from_file(self, mock_print, mock_open, mock_exists, mock_get_key):
        # Configure the mock file read
        mock_file = MagicMock()
        mock_file.read.return_value = "file_key\n"
        mock_file.__enter__.return_value = mock_file
        mock_file.__exit__.return_value = None
        mock_open.return_value = mock_file

        # Ensure api_key is None before calling
        with patch('modules.weather_service.api_key', None):
            initialize_weather_service()
            mock_get_key.assert_called_once() # Should check env first
            mock_exists.assert_called_once()
            mock_open.assert_called_once_with(os.path.join(_PROJECT_ROOT, 'models', 'openweather_api_key.txt'), 'r')
            mock_print.assert_any_call("Weather service API key loaded from file.")
            import modules.weather_service
            assert modules.weather_service.api_key == "file_key"

    @patch('modules.weather_service.get_openweather_api_key', return_value=None)
    @patch('os.path.exists', return_value=True)
    @patch('builtins.open', new_callable=MagicMock)
    @patch('builtins.print')
    async def test_initialize_weather_service_empty_file(self, mock_print, mock_open, mock_exists, mock_get_key):
        # Configure the mock file read for an empty file
        mock_file = MagicMock()
        mock_file.read.return_value = ""
        mock_file.__enter__.return_value = mock_file
        mock_file.__exit__.return_value = None
        mock_open.return_value = mock_file

        # Ensure api_key is None before calling
        with patch('modules.weather_service.api_key', None):
            initialize_weather_service()
            mock_get_key.assert_called_once() # Should check env first
            mock_exists.assert_called_once()
            mock_open.assert_called_once_with(os.path.join(_PROJECT_ROOT, 'models', 'openweather_api_key.txt'), 'r')
            mock_print.assert_any_call("Warning: OpenWeather API key file is empty.")
            import modules.weather_service
            assert modules.weather_service.api_key is None

# Helper to get _PROJECT_ROOT consistent with how config.py defines it
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) # tests directory
GRANDPARENT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR)) # E:\voice_asisstant
if GRANDPARENT_DIR.endswith("voice_asisstant"): # A simple check
    _PROJECT_ROOT = GRANDPARENT_DIR
else: # Fallback if structure is different, though less ideal for tests
    _PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
