# voice_assistant/core/tts.py
import pyttsx3
import threading
import queue
import time
import logging
from typing import Optional

class TTSEngine:
    """
    Threaded Text-to-Speech engine using pyttsx3.
    Supports queueing, interruption, and voice selection.
    """
    def __init__(self, voice_id: Optional[str] = None):
        """
        Initialize the TTS engine.
        Args:
            voice_id (str, optional): The ID of the voice to use. Defaults to None (system default).
        """
        self._engine = pyttsx3.init()
        self._lock = threading.Lock()
        self._speech_queue = queue.Queue()
        self._worker_thread = threading.Thread(target=self._process_queue, daemon=True)
        self._worker_thread.start()
        self._voice_cache = None
        if voice_id:
            try:
                self.set_voice(voice_id)
            except Exception as e_voice_set:
                logging.warning(f"TTS Engine: Could not set voice ID {voice_id}. Using default. Error: {e_voice_set}")
        else:
            logging.info("TTS Engine: No voice_id provided. Using default voice.")

    def _process_queue(self):
        """
        Processes the speech queue in a dedicated thread with retry logic.
        Retries up to 3 times on error before skipping text.
        """
        while True:
            text_to_speak = self._speech_queue.get()
            if text_to_speak is None:
                break
            with self._lock:
                for attempt in range(3):
                    try:
                        self._engine.say(text_to_speak)
                        self._engine.runAndWait()
                        break
                    except Exception as e:
                        logging.error(f"TTS Engine Error (attempt {attempt + 1}): {e}")
                        if attempt == 2:
                            logging.error("TTS Engine: Max retries reached. Skipping text.")
            time.sleep(0.1)

    def speak(self, text: str) -> None:
        """
        Adds text to the speech queue to be spoken by the TTS engine.
        This method is non-blocking.
        Args:
            text (str): The text to speak.
        """
        if text:
            self._speech_queue.put(text)

    def interrupt(self) -> None:
        """
        Interrupts ongoing speech and clears the queue.
        """
        with self._lock:
            self._engine.stop()
            while not self._speech_queue.empty():
                self._speech_queue.get()
            logging.info("TTS Engine: Speech interrupted and queue cleared.")

    def stop(self) -> None:
        """
        Stops the TTS worker thread gracefully and interrupts speech.
        """
        self.interrupt()
        self._speech_queue.put(None)
        self._worker_thread.join()

    def get_available_voices(self) -> list:
        """
        Returns a list of available TTS voices.
        Each voice in the list is a dictionary with 'id', 'name', and 'languages'.
        Returns:
            list: List of available voices.
        """
        if self._voice_cache is not None:
            return self._voice_cache
        available_voices = []
        try:
            raw_voices = self._engine.getProperty('voices')
            for voice in raw_voices:
                available_voices.append({
                    'id': voice.id,
                    'name': voice.name,
                    'languages': voice.languages
                })
            logging.info(f"TTS Engine: Found {len(available_voices)} available voices.")
            self._voice_cache = available_voices
        except Exception as e:
            logging.error(f"TTS Engine Error: Could not get available voices: {e}")
        return available_voices

    def set_voice(self, voice_id: str) -> bool:
        """
        Sets the TTS voice by its ID.
        Args:
            voice_id (str): The ID of the voice to set.
        Returns:
            bool: True if voice was set successfully, False otherwise.
        """
        try:
            current_voices = self.get_available_voices()
            if not any(v['id'] == voice_id for v in current_voices):
                logging.warning(f"TTS Engine: Voice ID '{voice_id}' not found among available voices. Voice not changed.")
                return False

            self._engine.setProperty('voice', voice_id)
            # Test the voice
            self._engine.say("Test")
            self._engine.runAndWait()
            logging.info(f"TTS Engine: Voice set to ID: {voice_id}")
            return True
        except Exception as e:
            logging.error(f"TTS Engine Error: Failed to set voice ID {voice_id}: {e}")
            return False

# Global instance of the TTS engine to be used across modules.
# This "singleton" approach ensures there's only one engine running.
tts_engine = TTSEngine()

# A simple function that modules can import.
def speak(text: str) -> None:
    """
    Convenience function for modules to call TTS.
    Args:
        text (str): The text to speak.
    """
    tts_engine.speak(text)

def stop_speech() -> None:
    """
    Interrupts ongoing speech and clears the queue.
    """
    tts_engine.interrupt()

# For future: TTS engine interface for multi-engine support
class TTSEngineInterface:
    """
    Interface for TTS engines. For future multi-engine support.
    """
    def say(self, text: str):
        """Speak the given text."""
        pass
    def run_and_wait(self):
        """Block until speaking is finished."""
        pass
    def set_voice(self, voice_id: str):
        """Set the voice by ID."""
        pass
