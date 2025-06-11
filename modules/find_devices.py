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
    Determines and returns the local IP address of the current machine.
    
    Attempts to establish a UDP connection to a public DNS server to infer the local IP address. Returns "Unknown" if the address cannot be determined.
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
    Scans the local network for active devices and announces the results via speech.
    
    Retrieves the local IP address, announces it, scans the network using ARP, logs each detected device, and provides spoken feedback on the number of devices found or any errors encountered.
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
    Returns a mapping of intent phrases to the corresponding handler function.
    
    The returned dictionary enables integration of the "find devices" and "scan network" intents with the main application's intent handling system.
    """
    return {
        "find devices": find_devices,
        "scan network": find_devices,
    }
