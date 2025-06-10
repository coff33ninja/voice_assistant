# voice_assistant/core/tts.py
import pyttsx3
import threading
import queue
import time

class TTSEngine:
    def __init__(self):
        self._engine = pyttsx3.init()
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

# Global instance of the TTS engine to be used across modules.
# This "singleton" approach ensures there's only one engine running.
tts_engine = TTSEngine()

# A simple function that modules can import.
def speak(text):
    """Convenience function for modules to call."""
    tts_engine.speak(text)
