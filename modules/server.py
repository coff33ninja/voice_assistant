"""
Module: server.py
Provides functions to boot and verify servers using Wake-on-LAN and ping.
"""

import os
import logging
import time
from typing import Optional
from core.tts import speak
from modules.wol import send_wol_packet
from modules.ping import ping_target
from modules.device_manager import get_device

CONFIG_PATH = os.path.join("modules", "configs", "systems_config.json")

def boot_system(system_name: str) -> None:
    """
    Boots the specified system using the device manager for config lookup.
    """
    device = get_device(system_name)
    if not device or "mac_address" not in device:
        response = f"MAC address for '{system_name}' is missing or device not found."
        logging.error(response)
        speak(response)
        return
    response_start = f"Sending WOL packet to '{system_name}'."
    logging.info(response_start)
    speak(response_start)
    success = send_wol_packet(str(device["mac_address"]))
    response_end = f"Boot command {'successful' if success else 'failed'} for '{system_name}'."
    logging.info(response_end)
    speak(response_end)

def start_server(server_name: Optional[str] = None) -> None:
    """
    Starts a server using Wake-on-LAN and attempts to verify its availability via ping, using device manager.
    """
    if not server_name:
        speak_msg = "Please specify which server you want to start. For example, say 'start server MyServerName'."
        logging.info("'start_server' called without server_name.")
        speak(speak_msg)
        return

    logging.info(f"Received request to start and verify server: {server_name}.")
    speak(f"Attempting to start and verify server {server_name}.")

    device = get_device(server_name)
    if not device or "mac_address" not in device:
        response = f"Server '{server_name}' not found or missing MAC address."
        logging.error(response)
        speak(response)
        return

    mac_address = device.get("mac_address")
    ip_address = device.get("ip_address")

    logging.info(f"Sending WOL packet to '{server_name}' ({mac_address}).")
    wol_success = send_wol_packet(str(mac_address))

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
