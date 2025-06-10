# voice_assistant/modules/server.py
from core.tts import speak
from modules.wol import load_systems_config, send_wol_packet

CONFIG_PATH = "systems_config.json"


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


def register_intents():
    """Returns a dictionary of intents to register with the main application."""
    return {
        "start server": start_server,
        "start the server": start_server,
        "boot system": boot_system,
    }
