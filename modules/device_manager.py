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
import platform
import subprocess
import threading
import time as _time

# Default config path for all modules
CONFIG_PATH = os.path.join("modules", "configs", "systems_config.json")

# Module-level cache for device configurations
_DEVICES_CACHE: Optional[Dict[str, Any]] = None
_CACHED_CONFIG_PATH: Optional[str] = None

# Constant for keys that have special handling in announcements
_ANNOUNCE_DEVICE_PRIMARY_KEYS = ("mac_address", "ip_address", "group", "type", "aliases")


def load_devices(config_path: str = CONFIG_PATH) -> Dict[str, Any]:
    """
    Loads devices from the systems configuration JSON file, using a cache.
    Returns an empty dict and provides TTS/log feedback on error.
    """
    global _DEVICES_CACHE, _CACHED_CONFIG_PATH

    if _DEVICES_CACHE is not None and _CACHED_CONFIG_PATH == config_path:
        return _DEVICES_CACHE.copy()  # Return a copy to prevent external modification

    try:
        # Ensure the directory exists before trying to open the file for reading
        config_dir = os.path.dirname(config_path)
        if config_dir: # Check if config_dir is not an empty string (e.g. if config_path is just a filename)
            os.makedirs(config_dir, exist_ok=True)

        with open(config_path, "r") as file:
            devices = json.load(file)
        _DEVICES_CACHE = devices
        _CACHED_CONFIG_PATH = config_path
        return devices
    except FileNotFoundError:
        logging.error(f"Device config not found at: {config_path}")
        speak(f"Device config not found at: {config_path}")
        _DEVICES_CACHE = {} # Initialize cache to empty
        _CACHED_CONFIG_PATH = config_path
        return {}
    except json.JSONDecodeError:
        logging.error(f"Invalid JSON in device config: {config_path}")
        speak(f"Invalid JSON in device config: {config_path}")
        _DEVICES_CACHE = {} # Initialize cache to empty
        _CACHED_CONFIG_PATH = config_path
        return {}


def get_device(name: str) -> Optional[Dict[str, Any]]:
    """
    Retrieves a device's info by name or alias from the config. Returns None if not found.
    Uses the cached device list.
    """
    # Uses the default CONFIG_PATH for loading, consistent with other functions
    # If a different config_path was intended for get_device, it would need to be passed to load_devices
    current_devices = load_devices()
    for dev_name, device in current_devices.items():
        if name.lower() == dev_name.lower():
            return device
        aliases = device.get("aliases", [])
        if isinstance(aliases, str):
            aliases = [aliases]
        if any(name.lower() == alias.lower() for alias in aliases):
            return device
    logging.warning(f"Device '{name}' not found in config.")
    speak(f"Device {name} not found in configuration.")
    return None


def _save_devices_and_update_cache(devices_to_save: Dict[str, Any], config_path: str = CONFIG_PATH) -> bool:
    """Helper to save devices to file and update the cache."""
    global _DEVICES_CACHE, _CACHED_CONFIG_PATH
    try:
        # Ensure the directory exists before trying to open the file for writing
        config_dir = os.path.dirname(config_path)
        if config_dir:
             os.makedirs(config_dir, exist_ok=True)
        with open(config_path, "w") as file:
            json.dump(devices_to_save, file, indent=4)
        _DEVICES_CACHE = devices_to_save.copy() # Update cache with a copy
        _CACHED_CONFIG_PATH = config_path
        return True
    except (IOError, OSError) as e:
        logging.error(f"Failed to save device config to {config_path}: {e}")
        speak("Failed to save device configuration.")
        return False


def list_devices() -> list:
    """
    Returns a list of all device names in the config.
    """
    devices = load_devices()
    return list(devices.keys())


def announce_device_info(device_name: str) -> None:
    """
    Announces the details (MAC, IP, etc.) for a specific device using TTS.
    """
    device = get_device(device_name)
    if not device:
        speak(f"Device {device_name} not found in configuration.")
        return
    details = []
    # Announce primary fields first, then others
    for key in _ANNOUNCE_DEVICE_PRIMARY_KEYS:
        if key in device and device[key]: # Check if key exists and has a value
            details.append(f"{key.replace('_', ' ').capitalize()}: {device[key]}")

    for key, value in device.items():
        if key not in _ANNOUNCE_DEVICE_PRIMARY_KEYS and value: # Announce other fields if they have a value
            details.append(f"{key.replace('_', ' ').capitalize()}: {value}")
    if details:
        speak(f"Device {device_name}: " + ", ".join(details))
    else:
        speak(f"No details found for device {device_name}.")


def add_device(name: str, mac_address: str = '', ip_address: str = '', group: str = '', type: str = '', **kwargs) -> None:
    """
    Adds a new device to the configuration. Overwrites if device with same name exists.
    """
    devices = load_devices()
    device_data = {
        k: v for k, v in {
            "mac_address": mac_address,
            "ip_address": ip_address,
            "group": group,
            "type": type # Parameter name 'type' matches key 'type'
        }.items() if v
    }
    device_data.update(kwargs) # Add any other dynamic fields
    devices[name] = device_data
    if _save_devices_and_update_cache(devices, CONFIG_PATH):
        speak(f"Device {name} added or updated successfully.")
    else:
        speak(f"Failed to add device {name}.")


def confirm_action(prompt: str) -> bool:
    """
    Asks the user for confirmation via TTS and input. Returns True if confirmed.
    """
    speak(prompt + " Please say or type 'yes' to confirm.")
    try:
        response = input(prompt + " (yes/no): ").strip().lower()
        return response == "yes"
    except Exception:
        return False


def remove_device(name: str) -> None:
    """
    Removes a device from the configuration.
    """
    devices = load_devices()
    if name in devices:
        if not confirm_action(f"Are you sure you want to remove device {name}?"):
            speak(f"Removal of device {name} cancelled.")
            return
        del devices[name]
        if _save_devices_and_update_cache(devices, CONFIG_PATH):
            speak(f"Device {name} removed successfully.")
        else:
            speak(f"Failed to remove device {name}.")
    else:
        speak(f"Device {name} not found in configuration.")


def update_device(name: str, **kwargs) -> None:
    """
    Updates fields for an existing device.
    """
    devices = load_devices()
    if name not in devices:
        speak(f"Device {name} not found in configuration.")
        return
    for k, v in kwargs.items():
        devices[name][k] = v
    if _save_devices_and_update_cache(devices, CONFIG_PATH):
        speak(f"Device {name} updated successfully.")
    else:
        speak(f"Failed to update device {name}.")


def _ping_ip(ip_address: str, timeout: int = 2) -> bool:
    """Pings an IP address and returns True if online, False otherwise."""
    if not ip_address:
        return False
    param = "-n" if platform.system().lower() == "windows" else "-c"
    # For Windows, timeout is -w <milliseconds>. For Linux/macOS, -W <seconds> or -t <seconds>.
    # Using subprocess.run's timeout parameter is more straightforward for overall command timeout.
    command = ["ping", param, "1", ip_address]
    try:
        subprocess.run(command, capture_output=True, text=True, check=True, timeout=timeout)
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False
    except FileNotFoundError: # Handle if ping command is not found
        logging.error("Ping command not found. Please ensure it's in your system's PATH.")
        return False

def check_all_device_statuses() -> None:
    """
    Pings all known devices and announces their online/offline status via TTS.
    """
    devices = load_devices()
    if not devices:
        speak("No devices found in configuration.")
        return
    results = []
    for name, info in devices.items():
        ip = info.get("ip_address")
        if not ip:
            results.append(f"{name}: IP not set")
            continue
        if _ping_ip(ip):
            results.append(f"{name}: online")
        else:
            results.append(f"{name}: offline")
    speak("; ".join(results))


def get_devices_by_group(group: str) -> dict:
    """
    Returns a dict of devices belonging to the specified group.
    """
    devices = load_devices()
    return {name: info for name, info in devices.items() if info.get("group", "").lower() == group.lower()}


def get_devices_by_type(type_filter: str) -> dict:
    """
    Returns a dict of devices matching the specified type.
    """
    devices = load_devices()
    return {name: info for name, info in devices.items() if info.get("type", "").lower() == type_filter.lower()}


def list_devices_by_type(type_: str) -> None:
    """
    Announces all device names of a given type.
    """
    devices = get_devices_by_type(type_)
    if devices:
        speak(f"Devices of type {type_}: {', '.join(devices.keys())}.")
    else:
        speak(f"No devices of type {type_} found.")


def wake_group(group: str) -> None:
    """
    Sends Wake-on-LAN packets to all devices in the specified group.
    """
    from modules.wol import send_wol_packet
    group_devices = get_devices_by_group(group)
    if not group_devices:
        speak(f"No devices found in group {group}.")
        return
    for name, info in group_devices.items():
        mac = info.get("mac_address")
        if mac:
            send_wol_packet(mac)
    speak(f"Wake-on-LAN packets sent to all devices in group {group}.")


def ping_group(group: str) -> None:
    """
    Pings all devices in the specified group and announces their status.
    """
    group_devices = get_devices_by_group(group)
    if not group_devices:
        speak(f"No devices found in group {group}.")
        return
    results = []
    for name, info in group_devices.items():
        ip = info.get("ip_address")
        if not ip:
            results.append(f"{name}: IP not set")
            continue
        if _ping_ip(ip):
            results.append(f"{name}: online")
        else:
            results.append(f"{name}: offline")
    speak(f"Group {group} status: " + "; ".join(results))


def register_intents() -> dict:
    """
    Returns intent mappings for device management (list, info, etc).
    """
    return {
        "list devices": lambda: speak(", ".join(list_devices())),
        "device info for": announce_device_info,
        "add device": add_device,
        "remove device": remove_device,
        "update device": update_device,
        "check all device statuses": check_all_device_statuses,
        "wake group": wake_group,
        "ping group": ping_group,
        "list devices by type": list_devices_by_type,
        # Add more device-related intents as needed
    }


def auto_discover_new_devices(interval_minutes: int = 10):
    """
    Periodically scans the network and notifies if new devices appear.
    Runs in a background thread.
    """
    from modules.find_devices import get_local_ip
    def discovery_loop():
        known_devices = load_devices()
        known_ips = {info.get("ip_address") for info in known_devices.values() if "ip_address" in info}
        local_ip = get_local_ip() # Get local IP once

        while True:
            try:
                result = subprocess.run(["arp", "-a"], capture_output=True, text=True, check=True)
                lines = result.stdout.splitlines()
                unknown_ips = []
                for line in lines:
                    stripped_line = line.strip()
                    if not stripped_line: # Skip empty lines
                        continue
                    # Skip header/interface lines and incomplete entries
                    if "Interface:" in stripped_line or "Internet Address" in stripped_line or "incomplete" in stripped_line.lower():
                        continue
                    parts = stripped_line.split()
                    if len(parts) >= 2:
                        ip_candidate = parts[0]
                        if ip_candidate != local_ip and \
                           not ip_candidate.startswith("224.") and \
                           not ip_candidate.endswith(".255") and \
                           ip_candidate.count('.') == 3:
                            if ip_candidate not in known_ips:
                                unknown_ips.append(ip_candidate)
                if unknown_ips:
                    speak(f"Automatic discovery: Found {len(unknown_ips)} new device IPs: {', '.join(unknown_ips)}.")
                _time.sleep(interval_minutes * 60)
            except Exception as e:
                logging.error(f"Auto-discovery error: {e}")
                _time.sleep(interval_minutes * 60)
    thread = threading.Thread(target=discovery_loop, daemon=True)
    thread.start()
    speak("Automatic device discovery started. You will be notified if new devices appear.")
