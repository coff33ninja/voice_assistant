"""
Module: weather.py
Fetches and speaks the current weather for a given city using Open-Meteo API.
"""

import os
import json
import requests
import logging
from core.tts import speak

# API URLs
GEOCODING_API_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_API_URL = "https://api.open-meteo.com/v1/forecast"

# WMO Weather interpretation codes (https://open-meteo.com/en/docs#weathervariables)
# This is a partial list for brevity; a more complete one can be added.
WMO_WEATHER_CODES = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snow fall",
    73: "Moderate snow fall",
    75: "Heavy snow fall",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm: Slight or moderate", # thunderstorm without hail
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail" # Only if specified in docs, otherwise 95 covers most
}

def get_weather_wmo_description(code: int) -> str:
    """
    Returns a human-readable description for the given WMO weather code.
    
    If the code is not recognized, returns "an unknown weather condition".
    """
    return WMO_WEATHER_CODES.get(code, "an unknown weather condition")

def get_weather(city_name: str) -> None:
    """
    Fetches and vocalizes the current weather for a specified city using the Open-Meteo API.
    
    If the city name is not provided, attempts to use a default city from the environment variable `DEFAULT_WEATHER_CITY`. Handles geocoding and weather retrieval errors by providing spoken feedback to the user.
    """
    # Use environment variable as fallback if city_name is empty
    if (not city_name or not city_name.strip()) and os.environ.get("DEFAULT_WEATHER_CITY"):
        city_name = os.environ["DEFAULT_WEATHER_CITY"]
        logging.info(f"No city provided, using default from environment: {city_name}")

    if not city_name or not city_name.strip():
        speak("You need to tell me a city name to get the weather for.")
        return

    logging.info(f"Getting weather for {city_name}...")
    speak(f"Getting weather for {city_name}.")

    lat, lon = None, None
    # --- 1. Geocode city name to latitude/longitude ---
    try:
        geo_params = {"name": city_name, "count": 1, "format": "json"}
        geo_response = requests.get(GEOCODING_API_URL, params=geo_params, timeout=5)
        geo_response.raise_for_status()
        geo_data = geo_response.json()

        if geo_data.get("results") and len(geo_data["results"]) > 0:
            location = geo_data["results"][0]
            lat = location.get("latitude")
            lon = location.get("longitude")
            actual_city_name = location.get("name", city_name) # Use name from API if available
            logging.info(f"Geocoded {city_name} to {actual_city_name} at ({lat}, {lon})")
        else:
            speak(f"Sorry, I couldn't find a location named {city_name}.")
            return

    except requests.exceptions.Timeout:
        logging.error(f"Timeout while geocoding city: {city_name}")
        speak("Sorry, the location lookup service timed out.")
        return
    except requests.exceptions.RequestException as e_geo:
        logging.error(f"Error geocoding city {city_name}: {e_geo}")
        speak(f"Sorry, I had trouble looking up the location for {city_name}.")
        return
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON response for geocoding {city_name}")
        speak("Sorry, I received an unexpected response when looking up the location.")
        return

    if lat is None or lon is None:
        # This case should ideally be caught by earlier checks, but as a safeguard:
        speak(f"Sorry, I couldn't get the coordinates for {city_name}.")
        return

    # --- 2. Get current weather using latitude/longitude ---
    try:
        weather_params = {
            "latitude": lat,
            "longitude": lon,
            "current_weather": True,
        }
        weather_response = requests.get(FORECAST_API_URL, params=weather_params, timeout=5)
        weather_response.raise_for_status()
        weather_data = weather_response.json()
        current = weather_data.get("current_weather", {})
        temp = current.get("temperature")
        code = current.get("weathercode")
        desc = get_weather_wmo_description(code)

        if temp is not None and desc:
            speak(f"The current weather in {city_name} is {desc} with a temperature of {temp} degrees Celsius.")
            logging.info(f"Weather for {city_name}: {desc}, {temp}°C")
        else:
            speak(f"Sorry, I couldn't get the weather details for {city_name}.")
    except requests.exceptions.Timeout:
        logging.error(f"Timeout while fetching weather for {city_name}")
        speak("Sorry, the weather service timed out.")
    except requests.exceptions.RequestException as e_weather:
        logging.error(f"Error fetching weather for {city_name}: {e_weather}")
        speak(f"Sorry, I had trouble getting the weather for {city_name}.")
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON response for weather in {city_name}")
        speak("Sorry, I received an unexpected response from the weather service.")

def register_intents() -> dict:
    """
    Returns a mapping of weather-related intent keywords to the get_weather function.
    
    This enables integration of weather query intents with the main application's intent handling system.
    """
    return {
        "get weather": get_weather,
        "weather in": get_weather,
    }
