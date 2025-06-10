# voice_assistant/modules/server.py
import os
import logging
import time
from core.tts import speak
from modules.wol import load_systems_config, send_wol_packet
from modules.ping import ping_target # For verifying server status after boot

# Path relative to project root, assuming script is run from project root
CONFIG_PATH = os.path.join("modules", "configs", "systems_config.json")


def boot_system(system_name):
    """Boot a system using Wake-on-LAN."""
    systems = load_systems_config(CONFIG_PATH)
    if system_name not in systems:
        response = f"System '{system_name}' not found in configuration."
        logging.error(response)
        speak(response)
        return

    mac_address = systems[system_name].get("mac_address")
    if not mac_address:
        response = f"MAC address for '{system_name}' is missing."
        logging.error(response)
        speak(response)
        return

    response_start = f"Sending WOL packet to '{system_name}'."
    logging.info(response_start)
    speak(response_start)

    success = send_wol_packet(mac_address)
    response_end = (
        f"Boot command {'successful' if success else 'failed'} for '{system_name}'."
    )
    logging.info(response_end)
    speak(response_end)

def start_server(server_name=None):
    """
    Starts a server by sending a Wake-on-LAN packet and then attempts to ping it
    to verify if it has come online.
    """
    if not server_name:
        speak_msg = "Please specify which server you want to start. For example, say 'start server MyServerName'."
        logging.info("INFO: 'start_server' called without server_name.")
        speak(speak_msg)
        return

    logging.info(f"ACTION: Received request to start and verify server: {server_name}.")
    speak(f"Attempting to start and verify server {server_name}.")

    systems = load_systems_config(CONFIG_PATH)
    if server_name not in systems:
        response = f"Server '{server_name}' not found in configuration."
        logging.error(response)
        speak(response)
        return

    system_config = systems[server_name]
    mac_address = system_config.get("mac_address")
    ip_address = system_config.get("ip_address")

    if not mac_address:
        response = f"MAC address for server '{server_name}' is missing. Cannot send Wake-on-LAN packet."
        logging.error(response)
        speak(response)
        return

    logging.info(f"Sending WOL packet to '{server_name}' ({mac_address}).")
    wol_success = send_wol_packet(mac_address)

    if wol_success:
        speak(f"Wake-on-LAN packet sent to {server_name}.")
        if ip_address:
            # Wait for the server to boot up before pinging
            # TODO: This delay could be configurable per system in systems_config.json
            boot_wait_time = 60 # seconds
            logging.info(f"Waiting {boot_wait_time} seconds for {server_name} to boot before pinging.")
            speak(f"I'll wait about a minute for {server_name} to boot, then I'll try to ping it.")
            time.sleep(boot_wait_time)

            logging.info(f"Attempting to ping {server_name} at {ip_address}.")
            # ping_target will speak its own results
            ping_target(server_name) # ping_target can resolve name to IP from config
        else:
            no_ip_response = f"{server_name} has been sent a boot command, but I cannot verify its status as its IP address is not configured."
            logging.warning(f"WARN: {no_ip_response}")
            speak(no_ip_response)
    else:
        response = f"Failed to send Wake-on-LAN packet to {server_name}."
        logging.error(response)
        speak(response)

# TODO: Add a 'stop_server(server_name)' function.
# This would require remote command execution (e.g., SSH) to gracefully shut down a server.
# This is a significant feature and out of scope for simple WOL/ping.
def register_intents():
    """Returns a dictionary of intents to register with the main application."""
    intents = {
        "start server": start_server,
        "start the server": start_server,
        # Intents that expect a system_name argument passed by main.py
        "boot system": boot_system,
        "turn on": boot_system,
        "wake": boot_system, # Short alias
    }
    # Note: The handle_command in main.py now supports extracting the argument after these phrases.
    # For example, "boot system MyPC" will call boot_system("MyPC").
    return intents
