# Python
import socket
import subprocess
from core.tts import speak

def find_devices():
    """Scans the local network for active devices."""
    print("ACTION: Scanning for devices on the network...")
    speak("Scanning for devices on the network.")

    try:
        result = subprocess.run(
            ["arp", "-a"], capture_output=True, text=True, check=True
        )
        devices = result.stdout.splitlines()
        for device in devices:
            print(device)
        speak(f"Found {len(devices)} devices on the network.")
    except Exception as e:
        print(f"ERROR: Failed to scan devices. {e}")
        speak("I encountered an error while scanning for devices.")

def register_intents():
    """Returns a dictionary of intents to register with the main application."""
    return {
        "find devices": find_devices,
        "scan network": find_devices,
    }
