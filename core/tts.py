# voice_assistant/core/tts.py
import pyttsx3
import threading
import queue
import time

class TTSEngine:
    def __init__(self, voice_id=None):
        self._engine = pyttsx3.init()
        if voice_id:
            try:
                self._engine.setProperty('voice', voice_id)
                print(f"TTS Engine: Successfully set voice to ID: {voice_id}")
            except Exception as e_voice_set:
                print(f"TTS Engine Warning: Could not set voice ID {voice_id}. Using default. Error: {e_voice_set}")
        else:
            try:
                print(f"TTS Engine: No voice_id provided. Using default voice.")
            except Exception as e_get_voice:
                print(f"TTS Engine: Using default voice. (Could not query current voice details: {e_get_voice})")
        self._lock = threading.Lock()
        self._speech_queue = queue.Queue()
        self._worker_thread = threading.Thread(target=self._process_queue, daemon=True)
        self._worker_thread.start()

    def _process_queue(self):
        """Processes the speech queue in a dedicated thread."""
        while True:
            text_to_speak = self._speech_queue.get()
            if text_to_speak is None: # Sentinel value to stop the thread
                break

            with self._lock:
                try:
                    self._engine.say(text_to_speak)
                    self._engine.runAndWait()
                except Exception as e:
                    print(f"TTS Engine Error: {e}")
            time.sleep(0.1) # Small delay to prevent tight loop issues

    def speak(self, text):
        """
        Adds text to the speech queue to be spoken by the TTS engine.
        This method is non-blocking.
        """
        if text:
            self._speech_queue.put(text)

    def stop(self):
        """Stops the TTS worker thread gracefully."""
        self._speech_queue.put(None)
        self._worker_thread.join()

    def get_available_voices(self):
        """Returns a list of available TTS voices.
        Each voice in the list is a dictionary with 'id', 'name', and 'languages'.
        """
        available_voices = []
        try:
            raw_voices = self._engine.getProperty('voices')
            for voice in raw_voices:
                available_voices.append({
                    'id': voice.id,
                    'name': voice.name,
                    'languages': voice.languages
                })
            print(f"TTS Engine: Found {len(available_voices)} available voices.")
        except Exception as e:
            print(f"TTS Engine Error: Could not get available voices: {e}")
        return available_voices

    def set_voice(self, voice_id):
        """Sets the TTS voice by its ID.
        Args:
            voice_id (str): The ID of the voice to set.
        Returns:
            bool: True if voice was set successfully, False otherwise.
        """
        try:
            current_voices = self.get_available_voices()
            if not any(v['id'] == voice_id for v in current_voices):
                print(f"TTS Engine Warning: Voice ID '{voice_id}' not found among available voices. Voice not changed.")
                return False

            self._engine.setProperty('voice', voice_id)
            print(f"TTS Engine: Voice successfully set to ID: {voice_id}")
            return True
        except Exception as e:
            print(f"TTS Engine Error: Could not set voice to ID {voice_id}: {e}")
            return False

# Global instance of the TTS engine to be used across modules.
# This "singleton" approach ensures there's only one engine running.
tts_engine = TTSEngine()

# A simple function that modules can import.
def speak(text):
    """Convenience function for modules to call."""
    tts_engine.speak(text)
