# voice_assistant/core/engine.py

import pvporcupine
import struct
import openwakeword
import pyaudio
import whisper
import audioop
import threading
import time
from collections import deque
import logging
from typing import Optional, List
import numpy as np


class VoiceCore:
    """
    VoiceCore handles wake word detection, audio streaming, and speech-to-text transcription.
    Supports OpenWakeWord, Picovoice, and STT-only modes.
    """

    def __init__(self, on_wake_word=None, on_command=None, whisper_model_name: str = "base.en", silence_threshold: int = 500, engine_type: str = "openwakeword", picovoice_access_key: Optional[str] = None, picovoice_keyword_paths: Optional[List[str]] = None, openwakeword_model_path: Optional[str] = None):
        """
        Initializes the Voice Core.

        Args:
            on_wake_word (function): Callback for wake word detection.
            on_command (function): Callback for transcribed command text.
            whisper_model_name (str): Whisper model to use.
            silence_threshold (int): RMS level for silence detection.
            engine_type (str): 'openwakeword', 'picovoice', or 'stt_only'.
            picovoice_access_key (str): Access key for Picovoice.
            picovoice_keyword_paths (list): Paths to Picovoice keyword model files.
            openwakeword_model_path (str): Path to OpenWakeWord model file.
        """
        self.on_wake_word = on_wake_word
        self.on_command = on_command
        self.silence_threshold = silence_threshold
        self.engine_type = engine_type
        self.picovoice_access_key = picovoice_access_key
        self.picovoice_keyword_paths = picovoice_keyword_paths or []
        self.openwakeword_model_path = openwakeword_model_path or ""
        self.porcupine = None
        self.oww = None # Explicitly initialize to None
        self.wakeword_model_name_oww = "" # Specific for openwakeword
        self._picovoice_audio_buffer = []

        logging.info(f"Core Engine: Initializing with engine_type: {self.engine_type}")
        if self.engine_type == "picovoice":
            if not self.picovoice_access_key:
                raise ValueError("Picovoice access key is required for 'picovoice' engine.")
            if not self.picovoice_keyword_paths or not isinstance(self.picovoice_keyword_paths, list):
                raise ValueError("Picovoice keyword paths (list) are required for 'picovoice' engine.")
            try:
                self.porcupine = pvporcupine.create(
                    access_key=self.picovoice_access_key,
                    keyword_paths=self.picovoice_keyword_paths
                )
                logging.info(f"Core Engine: Picovoice Porcupine initialized with models: {self.picovoice_keyword_paths}")
            except Exception as e:
                logging.error(f"Core Engine Error: Failed to initialize Picovoice Porcupine: {e}")
                raise # Re-raise the exception
        elif self.engine_type == "openwakeword":
            if not self.openwakeword_model_path:
                logging.info("INFO: openwakeword_model_path is None. Initializing in STT-only mode (no OWW engine loaded).")
                # self.oww will remain None
            else:
                try:
                    self.oww = openwakeword.Model(wakeword_models=[self.openwakeword_model_path])
                    self.wakeword_model_name_oww = self.openwakeword_model_path.split('/')[-1].replace('.onnx', '')
                    logging.info(f"Core Engine: openWakeWord initialized with model: {self.openwakeword_model_path}")
                except Exception as e:
                    logging.error(f"Core Engine Error: Failed to initialize openWakeWord: {e}")
                    raise # Re-raise the exception
        else:
            raise ValueError(f"Unsupported engine_type: {self.engine_type}")

        # --- Initialize Audio Stream (PyAudio) ---
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=1280, # Matches openWakeWord chunk size & Picovoice frame length
        )

        # --- Initialize Speech-to-Text Engine (Whisper) ---
        self.whisper_model = whisper.load_model(whisper_model_name)

        # --- State Management ---
        self.is_listening_for_command = False
        self.audio_buffer = deque(
            maxlen=15
        )  # Buffer to capture audio just before wake word
        self.command_audio = []
        self._stop_event = threading.Event()
        self.listen_thread = None

    def start(self) -> None:
        """
        Starts the main listening loop in a separate thread.
        """
        if self.listen_thread is None or not self.listen_thread.is_alive():
            logging.info("Core Engine: Starting...")
            self._stop_event.clear()
            self.listen_thread = threading.Thread(target=self._listen, daemon=True)
            self.listen_thread.start()
            logging.info("Core Engine: Listening for wake word...")

    def stop(self) -> None:
        """
        Stops the listening loop and cleans up resources. Idempotent.
        """
        if self._stop_event.is_set():
            return
        logging.info("Core Engine: Stopping...")
        self._stop_event.set()
        if self.listen_thread and self.listen_thread.is_alive():
            self.listen_thread.join()
        if hasattr(self, 'porcupine') and self.porcupine:
            logging.info("Core Engine: Releasing Picovoice Porcupine resources...")
            self.porcupine.delete()
            self.porcupine = None
        if self.stream.is_active():
            self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
        logging.info("Core Engine: Stopped gracefully.")

    def _restart_stream(self) -> None:
        """
        Restarts the PyAudio stream after repeated errors.
        """
        try:
            self.stream.stop_stream()
            self.stream.close()
        except Exception:
            pass
        self.stream = self.p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1280)
        logging.info("Core Engine: Audio stream restarted.")

    def _listen_openwakeword(self) -> None:
        """
        Main loop for OpenWakeWord engine. Handles wake word detection and error recovery.
        """
        error_count = 0
        while not self._stop_event.is_set():
            try:
                audio_chunk = self.stream.read(1280, exception_on_overflow=False)
                error_count = 0
            except IOError as e:
                error_count += 1
                logging.warning(f"Core Engine: Audio read error (OpenWakeWord): {e}")
                if error_count > 5:
                    self._restart_stream()
                    error_count = 0
                continue
            self.audio_buffer.append(audio_chunk)
            if self.oww is not None:
                # Convert audio_chunk (bytes) to numpy array for openwakeword
                audio_np = np.frombuffer(audio_chunk, dtype=np.int16)
                prediction = self.oww.predict(audio_np)
                # openwakeword.Model.predict returns a tuple: (scores, metadata)
                # If tuple, use first element for scores
                if isinstance(prediction, tuple):
                    scores = prediction[0]
                else:
                    scores = prediction
                # scores is a dict mapping model name to score
                if self.wakeword_model_name_oww and scores.get(self.wakeword_model_name_oww, 0) > 0.5:
                    self.is_listening_for_command = True
                    self.command_audio = list(self.audio_buffer)
                    if self.on_wake_word:
                        threading.Thread(target=self.on_wake_word).start()

    def _listen_picovoice(self) -> None:
        """
        Main loop for Picovoice engine. Handles wake word detection and error recovery.
        """
        error_count = 0
        while not self._stop_event.is_set():
            try:
                audio_chunk = self.stream.read(1280, exception_on_overflow=False)
                error_count = 0
            except IOError as e:
                error_count += 1
                logging.warning(f"Core Engine: Audio read error (Picovoice): {e}")
                if error_count > 5:
                    self._restart_stream()
                    error_count = 0
                continue
            try:
                current_samples = list(struct.unpack_from("h" * (len(audio_chunk) // 2), audio_chunk))
                self._picovoice_audio_buffer.extend(current_samples)
            except struct.error as se:
                logging.error(f"Core Engine: Failed to unpack audio chunk for Picovoice: {se}")
                continue
            if self.porcupine is not None:
                while len(self._picovoice_audio_buffer) >= self.porcupine.frame_length:
                    frame = self._picovoice_audio_buffer[:self.porcupine.frame_length]
                    del self._picovoice_audio_buffer[:self.porcupine.frame_length]
                    try:
                        keyword_index = self.porcupine.process(frame)
                        if keyword_index >= 0:
                            logging.info(f"Core Engine: Picovoice detected keyword (index {keyword_index}).")
                            self.is_listening_for_command = True
                            self.command_audio = [audio_chunk]
                            if self.on_wake_word:
                                threading.Thread(target=self.on_wake_word).start()
                            self._picovoice_audio_buffer = []
                            break
                    except pvporcupine.PorcupineActivationRefusedError as pare:
                        logging.warning(f"Core Engine: Picovoice Activation Refused: {pare}")
                        time.sleep(5)
                    except Exception as e_pv_process:
                        logging.error(f"Core Engine: Picovoice process() error: {e_pv_process}")
                        break

    def _listen_stt_only(self) -> None:
        """
        Main loop for STT-only mode. Always listens for commands.
        """
        error_count = 0
        while not self._stop_event.is_set():
            try:
                audio_chunk = self.stream.read(1280, exception_on_overflow=False)
                error_count = 0
            except IOError as e:
                error_count += 1
                logging.warning(f"Core Engine: Audio read error (STT-only): {e}")
                if error_count > 5:
                    self._restart_stream()
                    error_count = 0
                continue
            self.command_audio.append(audio_chunk)
            if self._is_silent(audio_chunk, self.silence_threshold):
                self._process_command()

    def _listen(self) -> None:
        """
        Dispatches to the appropriate listening loop based on engine type.
        """
        if self.engine_type == "openwakeword" and self.oww:
            self._listen_openwakeword()
        elif self.engine_type == "picovoice" and self.porcupine:
            self._listen_picovoice()
        elif self.engine_type == "openwakeword" and not self.oww:
            logging.info("Core Engine: Running in STT-only mode.")
            self._listen_stt_only()
        else:
            logging.error(f"Core Engine: Unknown or unsupported engine_type '{self.engine_type}' in _listen.")
            time.sleep(1)

    def _is_silent(self, data: bytes, threshold: int) -> bool:
        """
        Returns True if the audio chunk is below a volume threshold.

        Args:
            data (bytes): Audio data.
            threshold (int): RMS threshold.

        Returns:
            bool: True if silent, False otherwise.
        """
        return audioop.rms(data, 2) < threshold

    def _process_command(self) -> None:
        """
        Transcribes the recorded audio and triggers the command callback.

        Sensitive data (like API keys) is never logged.
        """
        import tempfile
        import os
        logging.info("Core Engine: Processing command...")
        # Save audio to a temporary file for Whisper
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp.write(b''.join(self.command_audio))
            tmp_path = tmp.name
        try:
            result = self.whisper_model.transcribe(tmp_path, fp16=False)
            command_text = result["text"]
            if isinstance(command_text, list):
                command_text = " ".join(str(x) for x in command_text)
            command_text = command_text.strip()
        finally:
            os.remove(tmp_path)
        self.command_audio = []
        self.is_listening_for_command = False
        logging.info("Core Engine: Listening for wake word...")
        if self.on_command and command_text:
            threading.Thread(target=self.on_command, args=(command_text,)).start()

    # For future: support multiple wake words and engine plugins
    # def add_wakeword_model(self, model_path: str):
    #     """Add a new wake word model at runtime."""
    #     pass
