# Python
import subprocess
from core.tts import speak

def ping_device(ip_address):
    """Ping a device to check its availability."""
    print(f"ACTION: Pinging {ip_address}...")
    speak(f"Pinging {ip_address}.")

    try:
        result = subprocess.run(
            ["ping", "-c", "4", ip_address], capture_output=True, text=True, check=True
        )
        print(result.stdout)
        speak(f"Ping to {ip_address} was successful.")
    except subprocess.CalledProcessError:
        print(f"ERROR: Ping to {ip_address} failed.")
        speak(f"Ping to {ip_address} failed.")
    except Exception as e:
        print(f"ERROR: Failed to ping {ip_address}. {e}")
        speak("I encountered an error while pinging the device.")

def register_intents():
    """Returns a dictionary of intents to register with the main application."""
    return {
        "ping device": ping_device,
        "ping server": ping_device,
    }
