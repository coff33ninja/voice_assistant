# voice_assistant/run_tests.py
import unittest
import os
import sys
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Add the project root to the Python path to allow importing from 'core'
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from core.tts import tts_engine, speak


def discover_tests():
    """Discover all tests in the 'tests' directory."""
    loader = unittest.TestLoader()
    return loader.discover(start_dir=os.path.join(os.path.dirname(__file__), "tests"))

def run_tests(suite):
    """Run the discovered test suite."""
    runner = unittest.TextTestRunner()
    return runner.run(suite)

def handle_test_results(result):
    """Handle the results of the test run."""
    if result.wasSuccessful():
        print("\n--- All tests passed successfully! ---")
        speak("All tests passed successfully!")
    else:
        print("\n--- Some tests failed. Please review the output. ---")
        speak("Some tests failed. Please review the console output for details.")

def run_all_tests():
    """Discovers and runs all tests in the 'tests' directory."""
    logger.info("Discovering and Running All Tests")
    speak("Starting system tests.")

    try:
        suite = discover_tests()
        result = run_tests(suite)
        handle_test_results(result)
    except FileNotFoundError:
        logger.error("'tests' directory not found.")
        speak("I could not find the tests directory.")
    finally:
        tts_engine.stop()


if __name__ == "__main__":
    run_all_tests()
