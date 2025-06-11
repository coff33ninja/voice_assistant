"""
Module: ping.py
Provides functions to ping devices by name or IP, using configuration for known systems.
"""

# Python
import subprocess
import platform
import os
import logging
from core.tts import speak
from modules.wol import load_systems_config as load_wol_systems_config # Use the one from wol.py

# Path relative to project root, assuming main.py is in project root
CONFIG_PATH = os.path.join("modules", "configs", "systems_config.json")

def _is_valid_ip_format(ip_string: str) -> bool:
    """
    Validates if the given string has a basic IPv4 format.
    Does not guarantee the IP is reachable or a valid public/private IP,
    only that it matches the X.X.X.X pattern with numbers in range.
    """
    parts = ip_string.split('.')
    if len(parts) != 4:
        return False
    for item in parts:
        if not item.isdigit():
            return False
        # Convert to int and check range
        try:
            if not 0 <= int(item) <= 255:
                return False
        except ValueError: # Should be caught by isdigit, but as a safeguard
            return False
    return True

def ping_target(target_identifier: str) -> None:
    """
    Attempts to ping a device specified by name or IP address, providing spoken feedback.
    
    If a device name is given, resolves it to an IP address using the systems configuration file.
    Executes a platform-appropriate ping command and announces the result via speech synthesis.
    Handles missing configuration, unknown devices, and various ping errors with appropriate user feedback.
    """
    systems = load_wol_systems_config(CONFIG_PATH) # Use the imported function
    if not systems: # load_wol_systems_config returns {} on error
        # Log message already handled by load_wol_systems_config, but we can add a ping-specific one
        logging.warning("Ping Module: Systems configuration could not be loaded or is empty. Ping by name might not work.")
        # We can still proceed if user provides an IP directly.
    ip_to_ping = None
    display_name = str(target_identifier) # Default to using the identifier itself for messages

    # Try to resolve if target_identifier is a name in config
    is_ip_address_format = _is_valid_ip_format(target_identifier)

    if not is_ip_address_format and systems: # If it's not IP-like and we have systems config
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
    elif is_ip_address_format:
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

# Alias for backward compatibility with tests and other modules
ping = ping_target

def register_intents() -> dict:
    """
    Registers and returns a mapping of intent phrases to the ping_target function.
    
    Returns:
        A dictionary where each key is an intent phrase and the value is the ping_target function, enabling the main application to handle ping-related commands.
    """
    return {
        # Intents that expect an argument (target name or IP) to be passed by main.py
        "ping": ping_target,
        "check status of": ping_target,
        "what is the status of": ping_target,
    }
