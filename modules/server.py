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
        # "boot system": boot_system, # TODO: Requires main.py to parse system_name from command
        # TODO: Generic intents like 'boot system' or 'ping target' require `main.py`'s
        #       `handle_command` to be enhanced to parse arguments (e.g., system name)
        #       from the user's spoken command and pass them to the action function.
    }
    # For now, to avoid errors, we only register intents that have a defined function
    # and do not require argument parsing from the command string that is not yet implemented.
    # If boot_system is to be used, it needs argument parsing in main.py's handle_command.
    # Example: if a user says "boot system MyPC", main.py needs to extract "MyPC".
    # Since that's not implemented, we can't safely register "boot system" as a generic phrase.
    # A more specific phrase that implies a direct call or a fixed target could work if needed.
    # Or, the function could prompt for the system_name if not provided. (Not implemented here)

    # Temporarily, let's add a test intent if needed or keep it empty to avoid issues.
    # For the purpose of this subtask, we ensure it returns a dict, even if empty.
    if 'boot_system' in globals() and callable(boot_system):
         # intents["boot my main pc"] = lambda: boot_system("MyMainPC") # Example of a specific variant
         pass # Not registering 'boot system' directly due to argument parsing needs.

    return intents
