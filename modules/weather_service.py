import asyncio # noqa F401
import aiohttp
import os
from typing import Optional, Tuple, Dict, Union, Any
from .config import get_openweather_api_key, OPENWEATHER_API_KEY_FILE_PATH
import logging

logger = logging.getLogger(__name__)

OPENWEATHER_API_URL = "https://api.openweathermap.org/data/2.5/weather"
IP_GEOLOCATION_URL = "https://ip-api.com/json/"
api_key = None

def initialize_weather_service():
    global api_key
    print("Initializing Weather service...")
    api_key = get_openweather_api_key()
    if api_key:
        print("Weather service API key loaded from environment.")
    elif os.path.exists(OPENWEATHER_API_KEY_FILE_PATH):
        with open(OPENWEATHER_API_KEY_FILE_PATH, "r") as f:
            api_key = f.read().strip()
        if api_key:
            print("Weather service API key loaded from file.")
        else:
            print("Warning: OpenWeather API key file is empty.")
            api_key = None # Ensure it's None if file was empty
    else:
        print(f"Warning: OpenWeather API key not found in environment or at {OPENWEATHER_API_KEY_FILE_PATH}. Weather service will not work.")
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
    entities: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]: # Ensure it always returns a Dict
    if not api_key:
        print("Error: OpenWeather API key not configured.")
        return {"error": "OpenWeather API key not configured."}

    params = {
        "appid": api_key,
        "units": "metric"  # Or "imperial" for Fahrenheit
    }
    actual_location_description_for_error = "the queried location" # Default
    coordinates_used: Optional[Tuple[float, float]] = None

    # 1. Handle date_reference entity for future forecasts
    if entities and entities.get("date_reference"):
        date_ref = str(entities.get("date_reference", "")).lower()
        # Simple check for "today". More robust parsing might be needed for other current-day references.
        is_today = "today" in date_ref or date_ref == "" # Empty might imply current from context

        # Check if it's not today and implies a specific future day (e.g., "tomorrow", "next tuesday")
        is_future_unsupported = not is_today and any(kw in date_ref for kw in ["tomorrow", "next", "on", "after", "evening", "morning", "pm", "am"])
        if is_future_unsupported:
            # A more sophisticated check could involve parsing date_ref with datetime utils
            # and comparing against current date if it's a specific date like "July 20th"
            # For now, broad keywords trigger the unsupported message.
            try:
                # Attempt to parse with reminder_utils helper if available and applicable
                # This is a placeholder for potential future integration if complex date parsing is needed here
                pass
            except Exception:
                pass # Ignore parsing errors for this check, rely on keywords

            if date_ref not in ["today", "now", "current"]: # Check if it's explicitly not for the current time
                 print(f"Forecast requested for '{entities.get('date_reference')}', but only current weather is supported.")
                 return {"message": f"I can only provide current weather, not forecasts for {entities.get('date_reference')}."}


    # 2. Determine location for the API call, prioritizing entities
    entity_location = str(entities.get("location")) if entities and entities.get("location") else None

    if entity_location:
        params["q"] = entity_location
        actual_location_description_for_error = entity_location
        print(f"Using location from entities: {entity_location}")
    elif isinstance(location_query, str): # Fallback to location_query if it's a string
        params["q"] = location_query
        actual_location_description_for_error = location_query
        print(f"Using location from location_query argument: {location_query}")
    elif isinstance(location_query, tuple) and len(location_query) == 2: # Lat/Lon from argument
        params["lat"] = str(location_query[0])
        params["lon"] = f"{location_query[1]:.4f}" # Format lon to 4 decimal places
        coordinates_used = location_query
        actual_location_description_for_error = f"coordinates (lat: {location_query[0]}, lon: {location_query[1]})"
        print(f"Using coordinates from location_query argument: {location_query}")
    elif location_query is None: # No specific location query, try IP geolocation
        print("No location provided via entities or arguments, attempting IP geolocation.")
        coordinates_used = await get_current_location_coordinates_async()
        if coordinates_used:
            params["lat"] = f"{coordinates_used[0]:.4f}" # Format lat
            params["lon"] = f"{coordinates_used[1]:.4f}" # Format lon
            actual_location_description_for_error = f"current location (lat: {coordinates_used[0]}, lon: {coordinates_used[1]})"
        else:
            print("Error: Could not determine current location for weather.")
            # Return a more specific message if IP geo fails but was the intended method
            return {"error": "Could not determine your current location for the weather."}
    else:
        print(f"Error: Invalid location_query type: {type(location_query)} and no usable entity found.")
        return {"error": "Invalid location query provided."}

    print(f"Requesting weather with params: {params}")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(OPENWEATHER_API_URL, params=params, raise_for_status=True) as response: # Use raise_for_status=True
                response.raise_for_status() # Raise an exception for HTTP errors
                data = await response.json()

                weather_list = data.get("weather")
                main_data = data.get("main")

                if weather_list and isinstance(weather_list, list) and len(weather_list) > 0 and \
                   main_data and isinstance(main_data, dict) and \
                   "description" in weather_list[0] and "temp" in main_data:

                    returned_city_name = data.get("name")
                    final_city_name = (
                        returned_city_name
                        if returned_city_name and returned_city_name.strip()
                        else None
                    )
                    # Determine the best city name to return
                    if not final_city_name: # If API didn't return a name or it was empty
                        if "q" in params: # We queried by name (from entity or location_query arg)
                            final_city_name = params["q"]
                        elif coordinates_used: # We queried by coordinates
                            if location_query is None: # IP Geo was used
                                final_city_name = f"your current area (around Lat {coordinates_used[0]:.2f}, Lon {coordinates_used[1]:.2f})" # type: ignore
                            else: # Coordinates were passed as argument
                                final_city_name = f"area at Lat {coordinates_used[0]:.2f}, Lon {coordinates_used[1]:.2f}" # type: ignore
                        else: # Should not happen if params were correctly set, but as a fallback
                            final_city_name = actual_location_description_for_error
                    return {
                        "description": weather_list[0]["description"],
                        "temp": main_data["temp"],
                        "city": final_city_name
                    }
                else:
                    logger.warning(f"Weather data for {actual_location_description_for_error} is malformed or missing key fields: {data}")
                    # Align with the test's expected error message structure for this case
                    location_desc_str = str(actual_location_description_for_error) if actual_location_description_for_error is not None else "an unspecified location"
                    return {"error": f"There was a problem fetching weather for {location_desc_str} (incomplete data)."}
    except aiohttp.ClientResponseError as e: # Catch specific HTTP errors first
        print(f"HTTP Error fetching weather for {actual_location_description_for_error}: {e.status} - {e.message}")
        # Check for 404 specifically for city not found, return a specific message.
        if e.status == 404 and ("q" in params):
            return {"error": f"Sorry, I couldn't find weather data for '{params['q']}'. Please check the location name."}
        elif e.status == 401:
            return {"error": "There was an authorization problem fetching weather (e.g., invalid API key)."}
        # Generic error for other client issues (e.g. 500) or if location was by coords.
        # Ensure actual_location_description_for_error is a string
        location_desc_str = str(actual_location_description_for_error) if actual_location_description_for_error is not None else "an unspecified location"
        return {"error": f"There was a problem fetching weather for {location_desc_str}. (HTTP Status: {e.status})"}
    except aiohttp.ClientConnectorError as e: # Catch network/connection errors
        print(f"Network Error fetching weather for {actual_location_description_for_error}: {e}")
        location_desc_str = str(actual_location_description_for_error) if actual_location_description_for_error is not None else "an unspecified location"
        return {"error": f"There was a network problem fetching weather for {location_desc_str}."}
    except aiohttp.ContentTypeError as e: # Explicitly catch ContentTypeError
        # This might happen if the response is not JSON, even if status is 200
        print(f"ContentTypeError when fetching weather for {actual_location_description_for_error}: {e}")
        return {"error": f"Received unexpected data format when fetching weather for {actual_location_description_for_error}."}
    except Exception as e:
        print(f"Unexpected error in get_weather_async for {actual_location_description_for_error}: {e}")
        # Generic error for truly unexpected issues
        location_desc_str = str(actual_location_description_for_error) if actual_location_description_for_error is not None else "an unspecified location"
        return {"error": f"An unexpected error occurred while fetching weather for {location_desc_str}."}
