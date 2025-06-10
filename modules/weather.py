import os
import json
import requests # For making HTTP requests
import logging
from core.tts import speak

# API URLs
GEOCODING_API_URL = "geocoding-api.open-meteo.com/v1/search"
FORECAST_API_URL = "api.open-meteo.com/v1/forecast"

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

def get_weather_wmo_description(code):
    return WMO_WEATHER_CODES.get(code, "an unknown weather condition")

def get_weather(city_name):
    """Fetches and speaks the current weather for a given city using Open-Meteo API."""
    if not city_name or not city_name.strip():
        speak("You need to tell me a city name to get the weather for.")
        return

    print(f"ACTION: Getting weather for {city_name}...")
    speak(f"Getting weather for {city_name}.")

    # --- 1. Geocode city name to latitude/longitude ---
    lat, lon = None, None
    try:
        geo_params = {"name": city_name, "count": 1, "format": "json"}
        # Ensure URL starts with http or https
        geo_url_to_use = GEOCODING_API_URL if GEOCODING_API_URL.startswith("http") else "https://" + GEOCODING_API_URL

        geo_response = requests.get(geo_url_to_use, params=geo_params, timeout=5)
        geo_response.raise_for_status()  # Raises an HTTPError for bad responses (4XX or 5XX)
        geo_data = geo_response.json()

        if geo_data.get("results") and len(geo_data["results"]) > 0:
            location = geo_data["results"][0]
            lat = location.get("latitude")
            lon = location.get("longitude")
            actual_city_name = location.get("name", city_name) # Use name from API if available
            print(f"Geocoded {city_name} to {actual_city_name} at ({lat}, {lon})")
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
            "current_weather": "true",
            # Optionally, specify temperature unit, e.g., "temperature_unit": "fahrenheit"
        }
        forecast_url_to_use = FORECAST_API_URL if FORECAST_API_URL.startswith("http") else "https://" + FORECAST_API_URL
        weather_response = requests.get(forecast_url_to_use, params=weather_params, timeout=10)
        weather_response.raise_for_status()
        weather_data = weather_response.json()

        if "current_weather" in weather_data:
            current = weather_data["current_weather"]
            temp = current.get("temperature")
            wind_speed = current.get("windspeed")
            weather_code = current.get("weathercode")

            # Get units from metadata (Open-Meteo provides this, good practice)
            temp_unit = weather_data.get("current_weather_units", {}).get("temperature", "°C")
            wind_unit = weather_data.get("current_weather_units", {}).get("windspeed", "km/h")

            weather_desc = get_weather_wmo_description(weather_code)

            response_parts = []
            if temp is not None:
                response_parts.append(f"a temperature of {temp}{temp_unit}")

            response_str = f"Currently in {actual_city_name}, it's {weather_desc}"
            if response_parts:
                response_str += ", with " + " and ".join(response_parts)

            if wind_speed is not None:
                response_str += f", and wind at {wind_speed} {wind_unit}."
            else:
                response_str += "."

            print(f"Weather for {actual_city_name}: Temp={temp}{temp_unit}, Wind={wind_speed}{wind_unit}, Desc='{weather_desc}' (Code: {weather_code})")
            speak(response_str)
        else:
            speak(f"Sorry, I couldn't get the current weather details for {actual_city_name}.")

    except requests.exceptions.Timeout:
        logging.error(f"Timeout while fetching weather for: {actual_city_name}")
        speak(f"Sorry, the weather service timed out for {actual_city_name}.")
    except requests.exceptions.RequestException as e_weather:
        logging.error(f"Error fetching weather for {actual_city_name}: {e_weather}")
        speak(f"Sorry, I had trouble getting the weather for {actual_city_name}.")
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON response for weather in {actual_city_name}")
        speak("Sorry, I received an unexpected response from the weather service.")

def register_intents():
    """Returns a dictionary of intents to register with the main application."""
    # These intents expect main.py to extract the city name and pass it as an argument.
    return {
        "what's the weather in": get_weather,
        "weather for": get_weather,
        "tell me the weather in": get_weather,
        "how is the weather in": get_weather,
        "temperature in": get_weather, # This will give full summary for now
        "weather": get_weather # A generic one, might need a default city or prompt if no city given
                               # For now, assumes city will be part of command, e.g. "weather london"
    }
