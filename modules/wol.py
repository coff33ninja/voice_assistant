# voice_assistant/modules/wol.py
import json
import socket


def load_systems_config(config_path):
    """Load systems configuration from a JSON file."""
    try:
        with open(config_path, "r") as file:
            systems = json.load(file)
        return systems
    except FileNotFoundError:
        print("ERROR: Configuration file not found.")
        return {}
    except json.JSONDecodeError:
        print("ERROR: Invalid JSON format.")
        return {}


def send_wol_packet(mac_address):
    """Send a Wake-on-LAN packet to the specified MAC address."""
    if len(mac_address) != 17:
        print("ERROR: Invalid MAC address format.")
        return False

    # Create the magic packet
    mac_bytes = bytes.fromhex(mac_address.replace(":", "").replace("-", ""))
    magic_packet = b"\xff" * 6 + mac_bytes * 16

    # Send the magic packet
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            sock.sendto(magic_packet, ("255.255.255.255", 9))
        print(f"ACTION: WOL packet sent to {mac_address}")
        return True
    except Exception as e:
        print(f"ERROR: Failed to send WOL packet. {e}")
        return False
