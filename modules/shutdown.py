"""
Module: shutdown.py
Provides functionality to shut down the computer with user confirmation.
"""

import os
import platform
import logging
import time
from core.tts import speak
from typing import Optional

def _perform_shutdown() -> None:
    """
    Actually executes the system shutdown command after confirmation.
    Provides spoken feedback and attempts to log the action.
    """
    system_os = platform.system().lower()
    logging.info(f"Shutdown confirmed. Attempting to shut down the system ({system_os}).")
    speak("Shutting down now. Goodbye!")

    # A small delay to allow TTS to finish, though it might be interrupted by the OS.
    time.sleep(2)

    try:
        if system_os == "windows":
            os.system("shutdown /s /f /t 1")  # Force close applications, shutdown in 1 sec
        elif system_os == "linux":
            # This command typically requires sudo privileges.
            # Ensure the user running the script has passwordless sudo for 'shutdown'
            # or is running as root. Alternatively, 'systemctl poweroff' can be used on systemd systems.
            logging.info("Executing Linux shutdown. This may require sudo privileges.")
            os.system("sudo shutdown -h now")
        elif system_os == "darwin":  # macOS
            # This command typically requires sudo privileges.
            # 'osascript -e \'tell app "System Events" to shut down\'' is an alternative
            # that might not require sudo but can be less forceful.
            logging.info("Executing macOS shutdown. This may require sudo privileges.")
            os.system("sudo shutdown -h now")
        else:
            logging.warning(f"Shutdown command not implemented for operating system: {system_os}")
            speak(f"Sorry, I don't know how to shut down a {system_os} system.")
    except Exception as e:
        logging.error(f"An error occurred while trying to execute shutdown command: {e}", exc_info=True)
        # Speak might not work if system is already shutting down, but try.
        speak("An error occurred trying to shut down the system.")

def request_shutdown_confirmation(argument: Optional[str] = None) -> None:
    """
    Asks the user for confirmation before initiating a system shutdown.
    The actual shutdown is triggered by a separate confirmation intent.
    The 'argument' parameter is accepted due to how main.py might pass it but is not used in this simple version.
    """
    logging.info(f"Shutdown requested (argument: {argument}). Asking for confirmation.")
    speak("Are you sure you want to shut down the computer? To confirm, please say 'yes confirm shutdown'.")

def register_intents() -> dict:
    """
    Registers intents for initiating and confirming system shutdown.
    
    Returns:
        A dictionary mapping intent phrases to their handler functions.
    """
    return {
        "shutdown computer": request_shutdown_confirmation,
        "shut down the computer": request_shutdown_confirmation,
        "turn off computer": request_shutdown_confirmation,
        "turn off the computer": request_shutdown_confirmation,
        "yes confirm shutdown": _perform_shutdown,  # This intent triggers the actual shutdown
    }