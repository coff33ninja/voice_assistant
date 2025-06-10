"""
Module: server.py
Provides functions to boot and verify servers using Wake-on-LAN and ping.
"""

import os
import logging
import time
from typing import Optional
from core.tts import speak
from modules.wol import load_systems_config, send_wol_packet
from modules.ping import ping_target

CONFIG_PATH = os.path.join("modules", "configs", "systems_config.json")

def boot_system(system_name: str) -> None:
    """
    Boot a system using Wake-on-LAN.
    Args:
        system_name (str): Name of the system to boot.
    """
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
    response_end = f"Boot command {'successful' if success else 'failed'} for '{system_name}'."
    logging.info(response_end)
    speak(response_end)

def start_server(server_name: Optional[str] = None) -> None:
    """
    Starts a server by sending a Wake-on-LAN packet and then attempts to ping it to verify if it has come online.
    Args:
        server_name (str, optional): Name of the server to start.
    """
    if not server_name:
        speak_msg = "Please specify which server you want to start. For example, say 'start server MyServerName'."
        logging.info("'start_server' called without server_name.")
        speak(speak_msg)
        return

    logging.info(f"Received request to start and verify server: {server_name}.")
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
            boot_wait_time = 60  # seconds
            logging.info(f"Waiting {boot_wait_time} seconds for {server_name} to boot before pinging.")
            speak(f"I'll wait about a minute for {server_name} to boot, then I'll try to ping it.")
            time.sleep(boot_wait_time)
            logging.info(f"Attempting to ping {server_name} at {ip_address}.")
            ping_target(server_name)
        else:
            no_ip_response = f"{server_name} has been sent a boot command, but I cannot verify its status as its IP address is not configured."
            logging.warning(no_ip_response)
            speak(no_ip_response)
    else:
        response = f"Failed to send Wake-on-LAN packet to {server_name}."
        logging.error(response)
        speak(response)

def stop_server() -> None:
    """
    Dummy implementation for stopping the server. To be implemented based on requirements.
    """
    speak("Stopping the server. This is a placeholder.")
    # Add actual stop logic here in the future
    pass

def register_intents() -> dict:
    """
    Returns a dictionary of intents to register with the main application.
    """
    return {
        "start server": start_server,
        "stop server": stop_server,
    }
