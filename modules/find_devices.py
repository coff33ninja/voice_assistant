"""
Module: find_devices.py
Scans the local network for active devices using ARP and provides voice feedback.
"""

import socket
import subprocess
import logging
from core.tts import speak

def get_local_ip() -> str:
    """
    Returns the local IP address of the current machine.
    """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception as e:
        logging.error(f"Could not determine local IP address: {e}")
        return "Unknown"

def find_devices() -> None:
    """
    Scans the local network for active devices and speaks the result.
    """
    local_ip = get_local_ip()
    logging.info(f"Local IP address: {local_ip}")
    speak(f"Your local IP address is {local_ip}. Scanning for devices on the network.")
    try:
        result = subprocess.run(["arp", "-a"], capture_output=True, text=True, check=True)
        devices = result.stdout.splitlines()
        for device in devices:
            logging.info(device)
        speak(f"Found {len(devices)} devices on the network.")
    except Exception as e:
        logging.error(f"Failed to scan devices: {e}")
        speak("I encountered an error while scanning for devices.")

def register_intents() -> dict:
    """
    Returns a dictionary of intents to register with the main application.
    """
    return {
        "find devices": find_devices,
        "scan network": find_devices,
    }
