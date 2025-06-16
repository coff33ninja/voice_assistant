import asyncio # noqa F401
import aiohttp
import os
from typing import Optional, Tuple, Dict, Union, Any
from .config import OPENWEATHER_API_KEY_FILE_PATH

OPENWEATHER_API_URL = "http://api.openweathermap.org/data/2.5/weather"
IP_GEOLOCATION_URL = "http://ip-api.com/json/"
api_key = None

def initialize_weather_service():
    global api_key
    print("Initializing Weather service...")
    if os.path.exists(OPENWEATHER_API_KEY_FILE_PATH):
        with open(OPENWEATHER_API_KEY_FILE_PATH, "r") as f:
            api_key = f.read().strip()
        if api_key:
            print("Weather service API key loaded.")
        else:
            print("Warning: OpenWeather API key file is empty.")
            api_key = None # Ensure it's None if file was empty
    else:
        print(f"Warning: OpenWeather API key file not found at {OPENWEATHER_API_KEY_FILE_PATH}. Weather service will not work.")
        api_key = None


async def get_current_location_coordinates_async() -> Optional[Tuple[float, float]]:
    """
    Attempts to get current latitude and longitude using IP geolocation.
    Returns (lat, lon) or None if an error occurs.
    """
    print("Attempting to get current location via IP geolocation...")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(IP_GEOLOCATION_URL) as response:
                response.raise_for_status()
                data = await response.json()
                if data.get("status") == "success" and "lat" in data and "lon" in data:
                    print(
                        f"IP Geolocation successful: Lat={data['lat']}, Lon={data['lon']}, City={data.get('city', 'N/A')}"
                    )
                    return (data["lat"], data["lon"])
                else:
                    print(
                        f"IP Geolocation failed or returned unexpected data: {data.get('message', 'No message')}"
                    )
                    return None
    except aiohttp.ClientError as e:
        print(f"Error fetching current location via IP: {e}")
    except Exception as e:
        print(f"Unexpected error in get_current_location_coordinates_async: {e}")
    return None


async def get_weather_async(
    location_query: Optional[Union[str, Tuple[float, float]]] = None,
) -> Optional[Dict[str, Any]]:
    if not api_key:
        print("Error: OpenWeather API key not configured.")
        return None

    params = {
        "appid": api_key,
        "units": "metric"  # Or "imperial" for Fahrenheit
    }

    coordinates_used: Optional[Tuple[float, float]] = None
    actual_location_description_for_error = "the queried location"

    if location_query is None:  # Request for current location
        coordinates_used = await get_current_location_coordinates_async()
        if coordinates_used:
            params["lat"] = str(coordinates_used[0])
            params["lon"] = str(coordinates_used[1])
            actual_location_description_for_error = f"current location (lat: {coordinates_used[0]}, lon: {coordinates_used[1]})"
        else:
            print("Error: Could not determine current location for weather.")
            return None
    elif (
        isinstance(location_query, tuple) and len(location_query) == 2
    ):  # Lat/Lon provided
        params["lat"] = str(location_query[0])
        params["lon"] = str(location_query[1])
        coordinates_used = location_query
        actual_location_description_for_error = (
            f"coordinates (lat: {location_query[0]}, lon: {location_query[1]})"
        )
    elif isinstance(location_query, str):  # City name provided
        params["q"] = location_query
        actual_location_description_for_error = location_query
    else:
        print(f"Error: Invalid location_query type: {type(location_query)}")
        return None

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(OPENWEATHER_API_URL, params=params) as response:
                response.raise_for_status() # Raise an exception for HTTP errors
                data = await response.json()
                if data.get("weather") and "main" in data:
                    returned_city_name = data.get("name")
                    final_city_name = (
                        returned_city_name
                        if returned_city_name and returned_city_name.strip()
                        else None
                    )

                    if not final_city_name:
                        if isinstance(location_query, str):
                            final_city_name = location_query
                        elif location_query is None and coordinates_used:
                            final_city_name = f"your current area (around Lat {coordinates_used[0]:.2f}, Lon {coordinates_used[1]:.2f})"
                        elif isinstance(location_query, tuple) and coordinates_used:
                            final_city_name = f"area at Lat {coordinates_used[0]:.2f}, Lon {coordinates_used[1]:.2f}"
                        else:
                            final_city_name = "the queried location"
                    return {
                        "description": data["weather"][0]["description"],
                        "temp": data["main"]["temp"],
                        "city": final_city_name
                    }
    except aiohttp.ClientError as e:
        print(f"Error fetching weather for {actual_location_description_for_error}: {e}")
    except Exception as e:
        print(f"Unexpected error in get_weather_async for {actual_location_description_for_error}: {e}")
    return None
