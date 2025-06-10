# voice_assistant/modules/wol.py
import json
import socket
import logging


def load_systems_config(config_path):
    """Load systems configuration from a JSON file."""
    try:
        with open(config_path, "r") as file:
            systems = json.load(file)
        return systems
    except FileNotFoundError:
        logging.error("Configuration file not found at path: %s", config_path)
        return {}
    except json.JSONDecodeError:
        logging.error("Invalid JSON format in configuration file: %s", config_path)
        return {}


def send_wol_packet(mac_address):
    """Send a Wake-on-LAN packet to the specified MAC address."""
    if len(mac_address) != 17:
        logging.error("Invalid MAC address format.")
        return False

    # Create the magic packet
    mac_bytes = bytes.fromhex(mac_address.replace(":", "").replace("-", ""))
    magic_packet = b"\xff" * 6 + mac_bytes * 16

    # Send the magic packet
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            sock.sendto(magic_packet, ("255.255.255.255", 9))
        logging.info(f"WOL packet sent to {mac_address}")
        return True
    except Exception as e:
        logging.error(f"Failed to send WOL packet to {mac_address}: {e}", exc_info=True)
        return False
