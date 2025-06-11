"""
Module: wol.py
Provides Wake-on-LAN functionality for network devices.
"""

import json
import socket
import logging
import re
import os
from typing import Dict, Any
from core.tts import speak

# Default config path for consistency with other modules
CONFIG_PATH = os.path.join("modules", "configs", "systems_config.json")


def is_valid_mac(mac: str) -> bool:
    """
    Validates a MAC address in the format XX:XX:XX:XX:XX:XX or XX-XX-XX-XX-XX-XX.
    """
    return bool(re.match(r"^([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})$", mac))


def load_systems_config(config_path: str = CONFIG_PATH) -> Dict[str, Any]:
    """
    Loads a systems configuration from a JSON file.
    
    Attempts to read and parse the specified file as JSON. If the file does not exist or contains invalid JSON, logs an error and returns an empty dictionary.
    
    Args:
        config_path: Path to the JSON configuration file.
    
    Returns:
        A dictionary representing the systems configuration, or an empty dictionary on error.
    """
    try:
        with open(config_path, "r") as file:
            systems = json.load(file)
        return systems
    except FileNotFoundError:
        logging.error(f"Configuration file not found at path: {config_path}")
        speak(f"Configuration file not found at path: {config_path}")
        return {}
    except json.JSONDecodeError:
        logging.error(f"Invalid JSON format in configuration file: {config_path}")
        speak(f"Invalid JSON format in configuration file: {config_path}")
        return {}


def send_wol_packet(mac_address: str, tts: bool = True) -> bool:
    """
    Sends a Wake-on-LAN magic packet to the specified MAC address.
    The MAC address must be in the format 'XX:XX:XX:XX:XX:XX' or 'XX-XX-XX-XX-XX-XX'.
    Returns True if the packet is sent successfully, or False if the MAC address is invalid or an error occurs.
    If tts is True, provides spoken feedback.
    """
    if not is_valid_mac(mac_address):
        logging.error("Invalid MAC address format.")
        if tts:
            speak("The MAC address format is invalid. Please provide a valid MAC address.")
        return False
    try:
        mac_bytes = bytes.fromhex(mac_address.replace(":", "").replace("-", ""))
        magic_packet = b"\xff" * 6 + mac_bytes * 16
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            sock.sendto(magic_packet, ("255.255.255.255", 9))
        logging.info(f"WOL packet sent to {mac_address}")
        if tts:
            speak(f"Wake on LAN packet sent to {mac_address}.")
        return True
    except Exception as e:
        logging.error(f"Failed to send WOL packet to {mac_address}: {e}", exc_info=True)
        if tts:
            speak(f"Failed to send Wake on LAN packet to {mac_address}.")
        return False


def wake_on_lan(mac_address: str) -> None:
    """
    Sends a Wake-on-LAN magic packet to the specified MAC address and provides spoken feedback.
    Args:
        mac_address: The MAC address of the device to wake, in the format 'XX:XX:XX:XX:XX:XX' or 'XX-XX-XX-XX-XX-XX'.
    """
    send_wol_packet(mac_address, tts=True)


def register_intents() -> dict:
    """
    Returns a mapping of intent strings to the wake-on-LAN handler function.
    
    This dictionary can be used to register WOL-related intents with an application,
    allowing both "wake on lan" and "send wol packet" to trigger the same handler.
    """
    return {
        "wake on lan": wake_on_lan,
        "send wol packet": wake_on_lan,
        "wake device": wake_on_lan,
    }
