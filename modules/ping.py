"""
Module: ping.py
Provides functions to ping devices by name or IP, using configuration for known systems.
"""

# Python
import subprocess
import platform
import os
import json
import logging
from core.tts import speak
from typing import Dict, Any

# Path relative to project root, assuming main.py is in project root
CONFIG_PATH = os.path.join("modules", "configs", "systems_config.json")

def load_systems_config() -> Dict[str, Any]:
    """
    Loads systems configuration for ping module.
    Returns:
        dict: Systems configuration dictionary.
    """
    if not os.path.exists(CONFIG_PATH):
        logging.error(f"Ping Module: Configuration file not found at {CONFIG_PATH}")
        return {}
    try:
        with open(CONFIG_PATH, "r") as file:
            systems = json.load(file)
        return systems
    except json.JSONDecodeError:
        logging.error(f"Ping Module: Invalid JSON format in {CONFIG_PATH}")
        return {}
    except Exception as e:
        logging.error(f"Ping Module: Error loading {CONFIG_PATH}: {e}")
        return {}

def ping_target(target_identifier: str) -> None:
    """
    Pings a device by its name (from config) or IP address and speaks the result.
    Args:
        target_identifier (str): Device name or IP address.
    """
    systems = load_systems_config()
    ip_to_ping = None
    display_name = str(target_identifier) # Default to using the identifier itself for messages

    # Try to resolve if target_identifier is a name in config
    # Basic IP address regex (very simplified) - does not validate all cases.
    is_ip_like = all(c.isdigit() or c == '.' for c in target_identifier) and target_identifier.count('.') == 3

    if not is_ip_like and systems: # If it's not IP-like and we have systems config
        if target_identifier in systems:
            if "ip_address" in systems[target_identifier] and systems[target_identifier]["ip_address"]:
                ip_to_ping = systems[target_identifier]["ip_address"]
                display_name = f"{target_identifier} ({ip_to_ping})" # e.g. PC1 (192.168.1.100)
                logging.info(f"Resolved '{target_identifier}' to IP '{ip_to_ping}'.")
            else:
                msg = f"Device '{target_identifier}' is in the configuration, but its IP address is missing."
                logging.error(msg)
                speak(msg)
                return
        else:
            msg = f"Sorry, I don't have a device named '{target_identifier}' in my configuration to ping."
            logging.warning(msg) # Changed to warning as it's a user input issue
            speak(msg)
            return
    elif is_ip_like:
        ip_to_ping = target_identifier
        # display_name is already target_identifier
    else: # Not IP-like and no systems config loaded or name not found (covered above)
        msg = f"Cannot resolve '{target_identifier}'. It does not look like an IP address and system configuration is unavailable or does not contain it."
        logging.error(msg)
        speak(msg)
        return

    if not ip_to_ping: # Should be caught by logic above, but as a safeguard
        logging.error(f"Could not determine IP address for target '{target_identifier}'.")
        speak(f"I could not determine the IP address for {display_name}.")
        return

    logging.info(f"Pinging {display_name}...")
    speak(f"Pinging {display_name}.")

    ping_command_params = ["-n", "4"] if platform.system().lower() == "windows" else ["-c", "4"]
    command = ["ping"] + ping_command_params + [ip_to_ping]

    try:
        # Using shell=False is safer, command is a list.
        result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=10) # Added timeout
        logging.info(f"Ping to {display_name} successful. Output:\n{result.stdout}")
        speak(f"Ping to {display_name} was successful.")
    except FileNotFoundError:
        err_msg_fnf = "The 'ping' command was not found on this system. I can't ping devices."
        logging.error(err_msg_fnf)
        speak(err_msg_fnf)
    except subprocess.CalledProcessError as cpe:
        # Ping failed (e.g., host unreachable)
        logging.warning(f"Ping to {display_name} failed. Command '{' '.join(cpe.cmd)}' returned {cpe.returncode}.")
        speak(f"Ping to {display_name} failed.")
    except subprocess.TimeoutExpired:
        logging.warning(f"Ping to {display_name} timed out.")
        speak(f"Ping to {display_name} timed out.")
    except Exception as e:
        # Catch any other unexpected errors during subprocess.run or elsewhere
        logging.error(f"An unexpected error occurred while pinging {display_name}: {e}", exc_info=True)
        speak(f"An unexpected error occurred while trying to ping {display_name}.")

def register_intents() -> dict:
    """
    Returns a dictionary of intents to register with the main application.
    """
    return {
        # Intents that expect an argument (target name or IP) to be passed by main.py
        "ping": ping_target,
        "check status of": ping_target,
        "what is the status of": ping_target,
    }
