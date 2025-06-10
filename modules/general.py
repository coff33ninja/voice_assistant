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
    States the current time using TTS.
    """
    current_time = time.strftime("%I:%M %p")
    response = f"The current time is {current_time}."
    logging.info(response)
    speak(response)


def run_self_test() -> None:
    """
    Runs the test suite and speaks the result.
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
    Logs a message and speaks it.

    Args:
        message (str): The message to log and speak.
        level (str): Logging level ('info' or 'error').
    """
    if level == "info":
        logging.info(message)
    elif level == "error":
        logging.error(message)
    speak(message)


def register_intents() -> dict:
    """
    Returns a dictionary of intents to register with the main application.
    """
    return {
        "what time is it": tell_time,
        "tell me the time": tell_time,
        "run self test": run_self_test,
        "run a self test": run_self_test,
    }
