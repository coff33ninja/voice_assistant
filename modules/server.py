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
    Sends a Wake-on-LAN packet to boot the specified system.
    
    If the system is not found or lacks a MAC address in the configuration, notifies the user and logs an error. Announces the boot attempt and its result via text-to-speech.
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
    Starts a server using Wake-on-LAN and attempts to verify its availability via ping.
    
    If no server name is provided, prompts the user to specify one. Loads the server configuration, sends a Wake-on-LAN packet to the server's MAC address, and, if an IP address is available, waits for the server to boot before attempting to ping it. Provides spoken and logged feedback for all key actions and error conditions.
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
    Announces that the server is being stopped. Placeholder for future stop logic.
    
    Currently, this function only provides a spoken message and does not perform any server shutdown actions.
    """
    speak("Stopping the server. This is a placeholder.")
    # Add actual stop logic here in the future
    pass

def register_intents() -> dict:
    """
    Returns a mapping of intent strings to their corresponding handler functions.
    
    This dictionary enables the main application to associate user intents with the appropriate server control functions.
    """
    return {
        "start server": start_server,
        "stop server": stop_server,
    }
