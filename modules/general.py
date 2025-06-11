"""
Module: general.py
Provides general utility actions such as telling the time and running self-tests.
"""

import time
from core.tts import speak
import subprocess
import sys
import logging


def tell_time() -> None:
    """
    Announces the current system time using text-to-speech.
    
    Retrieves the current time in hours and minutes with AM/PM, logs the message, and speaks it aloud.
    """
    current_time = time.strftime("%I:%M %p")
    response = f"The current time is {current_time}."
    logging.info(response)
    speak(response)


def run_self_test() -> None:
    """
    Runs the system self-test by executing the external test script.
    
    Initiates the `run_tests.py` script as a subprocess, logging and announcing the start of the test. If the script runs successfully, its output is logged. If an error occurs or the script is missing, logs the error and announces the failure.
    """
    log_and_speak("Running system self test now.")
    logging.info("Running self test...")
    try:
        process = subprocess.run(
            [sys.executable, "run_tests.py"], capture_output=True, text=True, check=True
        )
        logging.info(process.stdout)
        # The run_tests.py script will handle speaking the final result.
    except subprocess.CalledProcessError as e:
        logging.error("Test script encountered an error.")
        logging.error(e.stderr)
        log_and_speak("The self test script encountered an error.", level="error")
    except FileNotFoundError:
        logging.error("'run_tests.py' not found.")
        speak("I could not find the test runner script.")


def log_and_speak(message: str, level: str = "info") -> None:
    """
    Logs a message at the specified level and announces it using text-to-speech.
    
    Args:
        message: The message to log and speak.
        level: The logging level, either 'info' or 'error'.
    """
    if level == "info":
        logging.info(message)
    elif level == "error":
        logging.error(message)
    speak(message)


def hello() -> None:
    """
    Speaks a greeting message and prints a confirmation for the hello intent.
    
    Intended for testing the hello intent functionality.
    """
    speak("Hello! How can I help you?")
    print("ACTION: Hello intent triggered.")


def register_intents() -> dict:
    """
    Returns a mapping of intent strings to their corresponding handler functions.
    
    The returned dictionary enables the main application to associate user intents with the appropriate functions for handling greetings, time inquiries, and self-test commands.
    """
    return {
        "hello": hello,
        "what time is it": tell_time,
        "tell me the time": tell_time,
        "run self test": run_self_test,
        "run a self test": run_self_test,
    }
