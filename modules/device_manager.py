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
    Retrieves a device's info by name or alias from the config. Returns None if not found.
    """
    devices = load_devices(config_path)
    for dev_name, device in devices.items():
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


def list_devices(config_path: str = CONFIG_PATH) -> list:
    """
    Returns a list of all device names in the config.
    """
    devices = load_devices(config_path)
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
    if "mac_address" in device:
        details.append(f"MAC address: {device['mac_address']}")
    if "ip_address" in device:
        details.append(f"IP address: {device['ip_address']}")
    if "group" in device:
        details.append(f"Group: {device['group']}")
    if "type" in device:
        details.append(f"Type: {device['type']}")
    # Announce any other fields dynamically
    for key, value in device.items():
        if key not in ("mac_address", "ip_address", "group", "type"):
            details.append(f"{key.replace('_', ' ').capitalize()}: {value}")
    if details:
        speak(f"Device {device_name}: " + ", ".join(details))
    else:
        speak(f"No details found for device {device_name}.")


def add_device(name: str, mac_address: str = '', ip_address: str = '', group: str = '', type_: str = '', **kwargs) -> None:
    """
    Adds a new device to the configuration. Overwrites if device with same name exists.
    """
    devices = load_devices()
    device_data = {}
    if mac_address:
        device_data["mac_address"] = mac_address
    if ip_address:
        device_data["ip_address"] = ip_address
    if group:
        device_data["group"] = group
    if type_:
        device_data["type"] = type_
    # Add any extra fields
    for k, v in kwargs.items():
        device_data[k] = v
    devices[name] = device_data
    try:
        with open(CONFIG_PATH, "w") as file:
            json.dump(devices, file, indent=4)
        speak(f"Device {name} added or updated successfully.")
    except Exception as e:
        logging.error(f"Failed to add device {name}: {e}")
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
        try:
            with open(CONFIG_PATH, "w") as file:
                json.dump(devices, file, indent=4)
            speak(f"Device {name} removed successfully.")
        except Exception as e:
            logging.error(f"Failed to remove device {name}: {e}")
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
    try:
        with open(CONFIG_PATH, "w") as file:
            json.dump(devices, file, indent=4)
        speak(f"Device {name} updated successfully.")
    except Exception as e:
        logging.error(f"Failed to update device {name}: {e}")
        speak(f"Failed to update device {name}.")


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
        ping_command_params = ["-n", "1"] if platform.system().lower() == "windows" else ["-c", "1"]
        command = ["ping"] + ping_command_params + [ip]
        try:
            subprocess.run(command, capture_output=True, text=True, check=True, timeout=5)
            results.append(f"{name}: online")
        except Exception:
            results.append(f"{name}: offline")
    speak("; ".join(results))


def get_devices_by_group(group: str) -> dict:
    """
    Returns a dict of devices belonging to the specified group.
    """
    devices = load_devices()
    return {name: info for name, info in devices.items() if info.get("group", "").lower() == group.lower()}


def get_devices_by_type(type_: str) -> dict:
    """
    Returns a dict of devices matching the specified type.
    """
    devices = load_devices()
    return {name: info for name, info in devices.items() if info.get("type", "").lower() == type_.lower()}


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
        ping_command_params = ["-n", "1"] if platform.system().lower() == "windows" else ["-c", "1"]
        command = ["ping"] + ping_command_params + [ip]
        try:
            subprocess.run(command, capture_output=True, text=True, check=True, timeout=5)
            results.append(f"{name}: online")
        except Exception:
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
        while True:
            try:
                result = subprocess.run(["arp", "-a"], capture_output=True, text=True, check=True)
                lines = result.stdout.splitlines()
                unknown_ips = []
                local_ip = get_local_ip()
                for line in lines:
                    line = line.strip()
                    if not line or "Interface:" in line or "Internet Address" in line or not line.split() or "incomplete" in line.lower():
                        continue
                    parts = line.split()
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
