# voice_assistant/modules/server.py
import os
from core.tts import speak
from modules.wol import load_systems_config, send_wol_packet

# Path relative to project root, assuming script is run from project root
CONFIG_PATH = os.path.join("modules", "configs", "systems_config.json")


def boot_system(system_name):
    """Boot a system using Wake-on-LAN."""
    systems = load_systems_config(CONFIG_PATH)
    if system_name not in systems:
        response = f"System '{system_name}' not found in configuration."
        print(f"ERROR: {response}")
        speak(response)
        return

    mac_address = systems[system_name].get("mac_address")
    if not mac_address:
        response = f"MAC address for '{system_name}' is missing."
        print(f"ERROR: {response}")
        speak(response)
        return

    response_start = f"Sending WOL packet to '{system_name}'."
    print(f"ACTION: {response_start}")
    speak(response_start)

    success = send_wol_packet(mac_address)
    response_end = (
        f"Boot command {'successful' if success else 'failed'} for '{system_name}'."
    )
    print(f"ACTION: {response_end}")
    speak(response_end)

# def start_server():
# print("INFO: Placeholder for starting a server. Functionality not yet implemented.")
# speak("Server starting functionality is not yet implemented.")
# TODO: Implement actual server start logic if required.


def register_intents():
    """Returns a dictionary of intents to register with the main application."""
    intents = {
        # "start server": start_server, # Commented out as start_server is not defined
        # "start the server": start_server, # Commented out as start_server is not defined

        # Intents that expect a system_name argument passed by main.py
        "boot system": boot_system,
        "turn on": boot_system,
        "wake": boot_system, # Short alias
    }
    # Note: The handle_command in main.py now supports extracting the argument after these phrases.
    # For example, "boot system MyPC" will call boot_system("MyPC").
    return intents
