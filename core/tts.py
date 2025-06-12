"""
# voice_assistant/core/tts.py
Text-to-speech module with support for multiple engines (pyttsx3, gTTS).
Provides a threaded, asynchronous interface with voice selection, rate/pitch control, and interruption.
"""

import os
import logging
import threading
import queue
import time
from typing import Optional, List, Dict
from abc import ABC, abstractmethod
from core.user_config import load_config, save_config_type, DEFAULT_CONFIG
try:
    import pyttsx3
except ImportError:
    pyttsx3 = None
try:
    from gtts import gTTS
    import gTTS
    import pydub
    from pydub.playback import play
except ImportError:
    gTTS = None
    pydub = None
try:
    import pygame.mixer
except ImportError:
    pygame = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    filename='speech.log',
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class TTSEngineInterface(ABC):
    """Abstract interface for TTS engines."""
    @abstractmethod
    def say(self, text: str) -> None:
        """Queue text for asynchronous speech."""
        pass

    @abstractmethod
    def run_and_wait(self) -> None:
        """Block until all queued speech is complete."""
        pass

    @abstractmethod
    def set_voice(self, voice_id: str) -> bool:
        """Set the voice by ID."""
        pass

    @abstractmethod
    def get_available_voices(self) -> List[Dict]:
        """Get list of available voices."""
        pass

    @abstractmethod
    def set_rate(self, rate: float) -> bool:
        """Set speech rate (words per minute or multiplier)."""
        pass

    @abstractmethod
    def set_pitch(self, pitch: float) -> bool:
        """Set speech pitch (if supported)."""
        pass

    @abstractmethod
    def stop(self) -> None:
        """Stop speech and clean up resources."""
        pass

class Pyttsx3Engine(TTSEngineInterface):
    """TTS engine implementation using pyttsx3."""
    def __init__(self, voice_id: Optional[str] = None):
        if not pyttsx3:
            raise ImportError("pyttsx3 is not installed")
        self._engine = None
        self._lock = threading.Lock()
        self._speech_queue = queue.Queue()
        self._worker_thread = None
        self._voice_cache = None
        self._initialize_engine(voice_id)

    def _initialize_engine(self, voice_id: Optional[str]):
        """Initialize or reinitialize the pyttsx3 engine."""
        try:
            self._engine = pyttsx3.init()
            self._worker_thread = threading.Thread(target=self._process_queue, daemon=True)
            self._worker_thread.start()
            if voice_id:
                self.set_voice(voice_id)
            logging.info("Pyttsx3Engine initialized")
        except Exception as e:
            logging.error(f"Pyttsx3Engine init failed: {e}")
            raise

    def _process_queue(self):
        """Process queued text for speech."""
        while True:
            text = self._speech_queue.get()
            if text is None:
                break
            with self._lock:
                for attempt in range(3):
                    try:
                        self._engine.say(text)
                        self._engine.runAndWait()
                        break
                    except Exception as e:
                        logging.error(f"Pyttsx3Engine error (attempt {attempt + 1}): {e}")
                        if attempt == 2:
                            logging.error("Pyttsx3Engine: Max retries reached. Skipping text.")
                        else:
                            time.sleep(0.1)
                            self._initialize_engine(None)  # Reinitialize on failure
            time.sleep(0.1)

    def say(self, text: str) -> None:
        if text:
            self._speech_queue.put(text)

    def run_and_wait(self) -> None:
        with self._lock:
            self._engine.runAndWait()

    def set_voice(self, voice_id: str) -> bool:
        try:
            voices = self.get_available_voices()
            if not any(v['id'] == voice_id for v in voices):
                logging.warning(f"Pyttsx3Engine: Voice ID '{voice_id}' not found")
                return False
            self._engine.setProperty('voice', voice_id)
            self.say("Test")
            self.run_and_wait()
            logging.info(f"Pyttsx3Engine: Voice set to {voice_id}")
            return True
        except Exception as e:
            logging.error(f"Pyttsx3Engine: Failed to set voice {voice_id}: {e}")
            return False

    def get_available_voices(self) -> List[Dict]:
        if self._voice_cache:
            return self._voice_cache
        voices = []
        try:
            raw_voices = self._engine.getProperty('voices')
            for voice in raw_voices:
                voices.append({
                    'id': voice.id,
                    'name': voice.name,
                    'languages': voice.languages
                })
            self._voice_cache = voices
            logging.info(f"Pyttsx3Engine: Found {len(voices)} voices")
        except Exception as e:
            logging.error(f"Pyttsx3Engine: Failed to get voices: {e}")
        return voices

    def set_rate(self, rate: float) -> bool:
        try:
            self._engine.setProperty('rate', int(rate))
            logging.info(f"Pyttsx3Engine: Rate set to {rate}")
            return True
        except Exception as e:
            logging.error(f"Pyttsx3Engine: Failed to set rate {rate}: {e}")
            return False

    def set_pitch(self, pitch: float) -> bool:
        logging.warning("Pyttsx3Engine: Pitch adjustment not supported")
        return False

    def stop(self) -> None:
        with self._lock:
            self._engine.stop()
            while not self._speech_queue.empty():
                self._speech_queue.get()
            self._speech_queue.put(None)
            self._worker_thread.join()
            logging.info("Pyttsx3Engine stopped")

    def refresh_voice_cache(self) -> None:
        self._voice_cache = None
        self.get_available_voices()

class GTTSEngine(TTSEngineInterface):
    """TTS engine implementation using gTTS."""
    def __init__(self, voice_id: Optional[str] = None):
        if not gTTS or not pydub:
            raise ImportError("gTTS or pydub not installed")
        self._lock = threading.Lock()
        self._speech_queue = queue.Queue()
        self._worker_thread = threading.Thread(target=self._process_queue, daemon=True)
        self._worker_thread.start()
        self._cache_dir = "tts_cache"
        os.makedirs(self._cache_dir, exist_ok=True)
        self._lang = voice_id or 'en'
        self._rate = 1.0
        logging.info(f"GTTSEngine initialized with language: {self._lang}")

    def _process_queue(self):
        """Process queued text for speech."""
        pygame.mixer.init() if pygame else None
        while True:
            text = self._speech_queue.get()
            if text is None:
                break
            with self._lock:
                for attempt in range(3):
                    try:
                        cache_file = os.path.join(self._cache_dir, f"{hash(text)}_{self._lang}.mp3")
                        if not os.path.exists(cache_file):
                            tts = gTTS(text=text, lang=self._lang)
                            tts.save(cache_file)
                        audio = pydub.AudioSegment.from_mp3(cache_file)
                        if self._rate != 1.0:
                            audio = audio.speedup(playback_speed=self._rate)
                        play(audio)
                        break
                    except Exception as e:
                        logging.error(f"GTTSEngine error (attempt {attempt + 1}): {e}")
                        if attempt == 2:
                            logging.error("GTTSEngine: Max retries reached. Skipping text.")
            time.sleep(0.1)
        if pygame:
            pygame.mixer.quit()

    def say(self, text: str) -> None:
        if text:
            self._speech_queue.put(text)

    def run_and_wait(self) -> None:
        while not self._speech_queue.empty():
            time.sleep(0.1)

    def set_voice(self, voice_id: str) -> bool:
        try:
            self._lang = voice_id
            self.say("Test")
            self.run_and_wait()
            logging.info(f"GTTSEngine: Language set to {voice_id}")
            return True
        except Exception as e:
            logging.error(f"GTTSEngine: Failed to set language {voice_id}: {e}")
            return False

    def get_available_voices(self) -> List[Dict]:
        return [
            {'id': 'en', 'name': 'English', 'languages': ['en']},
            {'id': 'es', 'name': 'Spanish', 'languages': ['es']},
            {'id': 'fr', 'name': 'French', 'languages': ['fr']}
            # Add more as needed
        ]

    def set_rate(self, rate: float) -> bool:
        try:
            self._rate = rate
            logging.info(f"GTTSEngine: Rate set to {rate}")
            return True
        except Exception as e:
            logging.error(f"GTTSEngine: Failed to set rate {rate}: {e}")
            return False

    def set_pitch(self, pitch: float) -> bool:
        logging.warning("GTTSEngine: Pitch adjustment not supported")
        return False

    def stop(self) -> None:
        with self._lock:
            while not self._speech_queue.empty():
                self._speech_queue.get()
            self._speech_queue.put(None)
            self._worker_thread.join()
            if pygame and pygame.mixer.get_init():
                pygame.mixer.stop()
                pygame.mixer.quit()
            logging.info("GTTSEngine stopped")

class TTSEngineFactory:
    """Factory to create TTS engines based on configuration."""
    @staticmethod
    def create_engine(engine_type: str, voice_id: Optional[str] = None) -> TTSEngineInterface:
        if engine_type == "pyttsx3":
            return Pyttsx3Engine(voice_id)
        elif engine_type == "gtts":
            return GTTSEngine(voice_id)
        else:
            raise ValueError(f"Unsupported TTS engine: {engine_type}")

# Global TTS engine instance
config = load_config()
tts_engine_type = config.get("tts_engine_type", "pyttsx3")
tts_voice_id = config.get("chosen_tts_voice_id")
try:
    tts_engine = TTSEngineFactory.create_engine(tts_engine_type, tts_voice_id)
except Exception as e:
    logging.error(f"Failed to initialize TTS engine {tts_engine_type}: {e}")
    print(f"Error: TTS engine initialization failed. Falling back to pyttsx3. {e}")
    tts_engine = Pyttsx3Engine()

def speak(text: str) -> None:
    """Speak text using the global TTS engine."""
    tts_engine.say(text)

def stop_speech() -> None:
    """Stop ongoing speech."""
    tts_engine.stop()

def set_tts_rate(rate: float) -> bool:
    """Set speech rate."""
    return tts_engine.set_rate(rate)

def set_tts_pitch(pitch: float) -> bool:
    """Set speech pitch."""
    return tts_engine.set_pitch(pitch)
