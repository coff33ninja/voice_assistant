"""
Module: speedtest.py
Runs an internet speed test and provides results via TTS.
"""
import speedtest
import logging
from core.tts import speak

def run_speedtest() -> None:
    """
    Runs a speed test to measure internet speed and speaks the results.
    """
    logging.info("Running speed test...")
    speak("Running a speed test.")
    try:
        st = speedtest.Speedtest()
        st.get_best_server()
        download_speed = st.download() / 1_000_000  # Convert to Mbps
        upload_speed = st.upload() / 1_000_000  # Convert to Mbps
        logging.info(f"Download Speed: {download_speed:.2f} Mbps")
        logging.info(f"Upload Speed: {upload_speed:.2f} Mbps")
        speak(f"Download speed is {download_speed:.2f} Mbps.")
        speak(f"Upload speed is {upload_speed:.2f} Mbps.")
    except Exception as e:
        logging.error(f"Failed to run speed test: {e}")
        speak("I encountered an error while running the speed test.")

def register_intents() -> dict:
    """
    Returns a dictionary of intents to register with the main application.
    """
    return {
        "run speed test": run_speedtest,
        "check internet speed": run_speedtest,
    }
