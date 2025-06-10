# voice_assistant/modules/general.py
import time
from core.tts import speak
import subprocess
import sys
import logging


def tell_time():
    """Action function to state the current time."""
    current_time = time.strftime("%I:%M %p")
    response = f"The current time is {current_time}."
    print(f"ACTION: {response}")
    speak(response)


def run_self_test():
    """Action function to run the test suite."""
    log_and_speak("Running system self test now.")
    print("ACTION: Running self test...")

    # We use subprocess to run the test script and capture its output.
    # sys.executable ensures we use the same Python interpreter.
    try:
        process = subprocess.run(
            [sys.executable, "run_tests.py"], capture_output=True, text=True, check=True
        )
        print(process.stdout)  # Print the test output to the console
        # The run_tests.py script will handle speaking the final result.
    except subprocess.CalledProcessError as e:
        print("ACTION: Test script encountered an error.")
        print(e.stderr)
        log_and_speak("The self test script encountered an error.", level="error")
    except FileNotFoundError:
        print("ACTION: 'run_tests.py' not found.")
        speak("I could not find the test runner script.")


def log_and_speak(message, level="info"):
    """Logs a message and speaks it."""
    if level == "info":
        logging.info(message)
    elif level == "error":
        logging.error(message)
    speak(message)


def register_intents():
    """Returns a dictionary of intents to register with the main application."""
    return {
        "what time is it": tell_time,
        "tell me the time": tell_time,
        "run self test": run_self_test,
        "run a self test": run_self_test,
    }
