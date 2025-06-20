"""
Module: find_devices.py
Scans the local network for active devices using ARP and provides voice feedback.
"""

import socket
import subprocess
import logging
from core.tts import speak
from modules.device_manager import list_devices, load_devices

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
    Scans the local network for active devices and provides spoken feedback.
    
    Retrieves the local IP address, announces it, scans the network using ARP, and announces the number of devices found. If scanning fails, notifies the user via speech.
    """
    local_ip = get_local_ip()
    logging.info(f"Local IP address: {local_ip}")
    speak(f"Your local IP address is {local_ip}. Scanning for devices on the network.")
    try:
        result = subprocess.run(["arp", "-a"], capture_output=True, text=True, check=True)
        lines = result.stdout.splitlines()
        device_entries = 0
        parsed_devices_info = [] # For logging more structured info

        for line in lines:
            line = line.strip()
            if not line: # Skip empty lines
                continue
            
            # Basic filtering to avoid interface headers or invalid entries.
            # This is heuristic and might need adjustment based on OS output.
            if "Interface:" in line or "Internet Address" in line or not line.split() or "incomplete" in line.lower():
                logging.debug(f"Skipping ARP line: {line}")
                continue

            parts = line.split()
            # A common pattern is IP address followed by MAC address.
            # We are looking for lines that seem to contain at least an IP and a MAC.
            if len(parts) >= 2: 
                ip_candidate = parts[0]
                # A very basic check to see if it looks like an IP and not the local IP
                # and not a broadcast/multicast address.
                if ip_candidate != local_ip and \
                   not ip_candidate.startswith("224.") and \
                   not ip_candidate.endswith(".255") and \
                   ip_candidate.count('.') == 3: # Simple IP format check
                    device_entries += 1
                    parsed_devices_info.append(f"Potential device: {' '.join(parts)}")

        for info in parsed_devices_info:
            logging.info(info)
        
        speak(f"Found approximately {device_entries} other devices on the network.")
    except Exception as e:
        logging.error(f"Failed to scan devices: {e}")
        speak("I encountered an error while scanning for devices.")

def announce_known_devices() -> None:
    """
    Announces the list of known devices from the configuration using TTS.
    """
    devices = list_devices()
    if devices:
        speak(f"Known devices are: {', '.join(devices)}.")
    else:
        speak("No known devices found in the configuration.")

def find_and_suggest_new_devices() -> None:
    """
    Scans the network, compares ARP results to known devices, and suggests adding unknown ones.
    """
    local_ip = get_local_ip()
    known_devices = load_devices()
    known_ips = {info.get("ip_address") for info in known_devices.values() if "ip_address" in info}
    try:
        result = subprocess.run(["arp", "-a"], capture_output=True, text=True, check=True)
        lines = result.stdout.splitlines()
        unknown_ips = []
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
            speak(f"Found {len(unknown_ips)} unknown device IPs: {', '.join(unknown_ips)}. You may want to add them to your configuration.")
        else:
            speak("No unknown devices found on the network.")
    except Exception as e:
        logging.error(f"Failed to scan devices: {e}")
        speak("I encountered an error while scanning for devices.")

def register_intents() -> dict:
    """
    Returns a mapping of intent phrases to the corresponding handler function.
    
    The returned dictionary enables integration with an intent-based system, allowing the phrases "find devices" and "scan network" to trigger the device scanning functionality.
    """
    return {
        "find devices": find_devices,
        "scan network": find_devices,
        "list known devices": announce_known_devices,
        "find and suggest new devices": find_and_suggest_new_devices,
    }
