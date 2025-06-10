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
    Load systems configuration from a JSON file.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        dict: Systems configuration dictionary.
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
    Send a Wake-on-LAN packet to the specified MAC address.

    Args:
        mac_address (str): MAC address in format 'XX:XX:XX:XX:XX:XX'.

    Returns:
        bool: True if packet sent successfully, False otherwise.
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
