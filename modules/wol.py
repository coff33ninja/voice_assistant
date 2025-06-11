"""
Module: wol.py
Provides Wake-on-LAN functionality for network devices.
"""

import json
import socket
import logging
from typing import Dict, Any


def load_systems_config(config_path: str) -> Dict[str, Any]:
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
        return {}
    except json.JSONDecodeError:
        logging.error(f"Invalid JSON format in configuration file: {config_path}")
        return {}


def send_wol_packet(mac_address: str) -> bool:
    """
    Sends a Wake-on-LAN magic packet to the specified MAC address.
    
    Args:
        mac_address: The target device's MAC address in the format 'XX:XX:XX:XX:XX:XX'.
    
    Returns:
        True if the magic packet was sent successfully, False otherwise.
    """
    if len(mac_address) != 17:
        logging.error("Invalid MAC address format.")
        return False
    try:
        mac_bytes = bytes.fromhex(mac_address.replace(":", "").replace("-", ""))
        magic_packet = b"\xff" * 6 + mac_bytes * 16
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            sock.sendto(magic_packet, ("255.255.255.255", 9))
        logging.info(f"WOL packet sent to {mac_address}")
        return True
    except Exception as e:
        logging.error(f"Failed to send WOL packet to {mac_address}: {e}", exc_info=True)
        return False


def wake_on_lan(mac_address: str) -> bool:
    """
    Sends a Wake-on-LAN magic packet to the specified MAC address.
    
    This function is an alias for `send_wol_packet`, provided for backward compatibility.
    
    Args:
        mac_address: The MAC address of the device to wake, in the format XX:XX:XX:XX:XX:XX.
    
    Returns:
        True if the magic packet was sent successfully, False otherwise.
    """
    return send_wol_packet(mac_address)


def register_intents() -> dict:
    """
    Returns a mapping of intent strings to the wake-on-LAN handler function.
    
    This dictionary can be used to integrate Wake-on-LAN functionality with an application's intent handling system.
    """
    return {
        "wake on lan": wake_on_lan,
        "send wol packet": wake_on_lan,
    }
