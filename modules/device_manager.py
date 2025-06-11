"""
Module: device_manager.py
Centralized device manager for loading, validating, and retrieving device info from systems_config.json.
Provides reusable functions for other modules and intent handlers.
"""

import os
import json
import logging
from typing import Dict, Any, Optional
from core.tts import speak

# Default config path for all modules
CONFIG_PATH = os.path.join("modules", "configs", "systems_config.json")


def load_devices(config_path: str = CONFIG_PATH) -> Dict[str, Any]:
    """
    Loads devices from the systems configuration JSON file.
    Returns an empty dict and provides TTS/log feedback on error.
    """
    try:
        with open(config_path, "r") as file:
            devices = json.load(file)
        return devices
    except FileNotFoundError:
        logging.error(f"Device config not found at: {config_path}")
        speak(f"Device config not found at: {config_path}")
        return {}
    except json.JSONDecodeError:
        logging.error(f"Invalid JSON in device config: {config_path}")
        speak(f"Invalid JSON in device config: {config_path}")
        return {}


def get_device(name: str, config_path: str = CONFIG_PATH) -> Optional[Dict[str, Any]]:
    """
    Retrieves a device's info by name from the config. Returns None if not found.
    """
    devices = load_devices(config_path)
    device = devices.get(name)
    if not device:
        logging.warning(f"Device '{name}' not found in config.")
        speak(f"Device {name} not found in configuration.")
        return None
    return device


def list_devices(config_path: str = CONFIG_PATH) -> list:
    """
    Returns a list of all device names in the config.
    """
    devices = load_devices(config_path)
    return list(devices.keys())


def register_intents() -> dict:
    """
    Returns intent mappings for device management (list, info, etc).
    """
    return {
        "list devices": lambda: speak(", ".join(list_devices())),
        # Add more device-related intents as needed
    }
