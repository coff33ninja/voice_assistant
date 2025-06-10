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


class VoiceCore:
    def __init__(self, on_wake_word=None, on_command=None, whisper_model_name="base.en", silence_threshold=500, engine_type="openwakeword", picovoice_access_key=None, picovoice_keyword_paths=None, openwakeword_model_path=None):
        """
        Initializes the Voice Core.

        Args:
            on_wake_word (function): A callback function to execute when the wake word is detected.
            on_command (function): A callback function to execute with the transcribed command text.
            whisper_model_name (str): Name of the Whisper model to use.
            silence_threshold (int): RMS level for silence detection.
            engine_type (str): 'openwakeword' or 'picovoice'.
            picovoice_access_key (str): Access key for Picovoice.
            picovoice_keyword_paths (list): List of paths to Picovoice keyword model files.
            openwakeword_model_path (str): Path to the OpenWakeWord model file.
        """
        self.on_wake_word = on_wake_word
        self.on_command = on_command
        self.silence_threshold = silence_threshold
        self.engine_type = engine_type
        self.picovoice_access_key = picovoice_access_key
        self.picovoice_keyword_paths = picovoice_keyword_paths
        self.openwakeword_model_path = openwakeword_model_path
        self.porcupine = None
        self.oww = None # Explicitly initialize to None
        self.wakeword_model_name_oww = "" # Specific for openwakeword
        self._picovoice_audio_buffer = []

        print(f"Core Engine: Initializing with engine_type: {self.engine_type}")
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
                print(f"Core Engine: Picovoice Porcupine initialized with models: {self.picovoice_keyword_paths}")
            except Exception as e:
                print(f"Core Engine Error: Failed to initialize Picovoice Porcupine: {e}")
                raise # Re-raise the exception
        elif self.engine_type == "openwakeword":
            if not self.openwakeword_model_path:
                print("INFO: openwakeword_model_path is None. Initializing in STT-only mode (no OWW engine loaded).")
                # self.oww will remain None
            else:
                try:
                    self.oww = openwakeword.Model(wakeword_models=[self.openwakeword_model_path])
                    self.wakeword_model_name_oww = self.openwakeword_model_path.split('/')[-1].replace('.onnx', '')
                    print(f"Core Engine: openWakeWord initialized with model: {self.openwakeword_model_path}")
                except Exception as e:
                    print(f"Core Engine Error: Failed to initialize openWakeWord: {e}")
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

    def start(self):
        """Starts the main listening loop in a separate thread."""
        if self.listen_thread is None or not self.listen_thread.is_alive():
            print("Core Engine: Starting...")
            self._stop_event.clear()
            self.listen_thread = threading.Thread(target=self._listen, daemon=True)
            self.listen_thread.start()
            print(f"Core Engine: Listening for wake word...")

    def stop(self):
        """Stops the listening loop and cleans up resources."""
        print("Core Engine: Stopping...")
        self._stop_event.set()
        if self.listen_thread:
            self.listen_thread.join()  # Wait for the thread to finish

        if hasattr(self, 'porcupine') and self.porcupine: # Check attribute existence too
            print("Core Engine: Releasing Picovoice Porcupine resources...")
            self.porcupine.delete()
            self.porcupine = None

        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
        print("Core Engine: Stopped gracefully.")

    def _listen(self):
        """The main loop that processes audio chunks, handling both OpenWakeWord and Picovoice."""
        # Ensure _picovoice_audio_buffer is initialized in __init__; e.g., self._picovoice_audio_buffer = []
        while not self._stop_event.is_set():
            try:
                # PyAudio stream configured for 1280 frames per read in __init__
                audio_chunk_bytes = self.stream.read(1280, exception_on_overflow=False)
            except IOError as e:
                if "Input overflowed" in str(e): # Check string for wider compatibility
                    print("Core Engine Warning: Input overflowed. Dropping frame.")
                    continue
                print(f"Core Engine Error: PyAudio read error: {e}")
                time.sleep(0.1)  # Avoid tight loop on persistent error
                continue # Or break, depending on desired robustness for other IOErrors

            if self.is_listening_for_command:
                self.command_audio.append(audio_chunk_bytes)
                if self._is_silent(audio_chunk_bytes, threshold=self.silence_threshold):
                    self._process_command()
            else: # Not listening for command, so in wake word detection mode
                if self.engine_type == "openwakeword":
                    if not self.oww: # oww should be initialized if this engine_type is selected
                        print("Core Engine Error: openWakeWord engine (self.oww) is not initialized!")
                        time.sleep(0.1); continue
                    # audio_buffer is used by openwakeword to capture audio just before wake word
                    self.audio_buffer.append(audio_chunk_bytes)
                    prediction = self.oww.predict(audio_chunk_bytes)
                    if self.wakeword_model_name_oww and prediction.get(self.wakeword_model_name_oww, 0) > 0.5: # Ensure correct attribute name is used
                        self.is_listening_for_command = True
                        self.command_audio = list(self.audio_buffer) # Copy pre-wake-word audio
                        if self.on_wake_word:
                            threading.Thread(target=self.on_wake_word).start()
                elif self.engine_type == "picovoice":
                    if not self.porcupine: # porcupine should be initialized
                        print("Core Engine Error: Picovoice engine (self.porcupine) is not initialized!")
                        time.sleep(0.1); continue
                    try:
                        # Convert raw byte chunk to int16 samples
                        current_samples = list(struct.unpack_from("h" * (len(audio_chunk_bytes) // 2), audio_chunk_bytes))
                        if not hasattr(self, '_picovoice_audio_buffer'): self._picovoice_audio_buffer = [] # Defensive init
                        self._picovoice_audio_buffer.extend(current_samples)
                    except struct.error as se:
                        print(f"Core Engine Error: Failed to unpack audio chunk for Picovoice: {se}. Chunk len: {len(audio_chunk_bytes)}")
                        continue # Skip this corrupted or unexpectedly short chunk

                    # Process available audio in porcupine.frame_length (e.g., 512 samples) chunks
                    while len(self._picovoice_audio_buffer) >= self.porcupine.frame_length:
                        frame_to_process = self._picovoice_audio_buffer[:self.porcupine.frame_length]
                        del self._picovoice_audio_buffer[:self.porcupine.frame_length]
                        try:
                            keyword_index = self.porcupine.process(frame_to_process)
                            if keyword_index >= 0:
                                print(f"Core Engine: Picovoice detected keyword (index {keyword_index}).")
                                self.is_listening_for_command = True
                                # For Picovoice, command audio typically starts *after* the wake word.
                                # The current audio_chunk_bytes contains the end of the wake word.
                                # self._picovoice_audio_buffer might contain some audio immediately after.
                                # Decide if we want to include buffered audio or start fresh.
                                self.command_audio = [audio_chunk_bytes] # Start with the raw chunk that triggered it
                                if self.on_wake_word:
                                    threading.Thread(target=self.on_wake_word).start()
                                self._picovoice_audio_buffer = [] # Clear buffer after detection
                                break  # Exit from 'while len >= frame_length' loop, process this detection
                        except pvporcupine.PorcupineActivationRefusedError as pare:
                            print(f"Core Engine Warning: Picovoice Activation Refused. Check AccessKey limits/status: {pare}")
                            time.sleep(5) # Pause significantly if this happens
                        except Exception as e_pv_process:
                            print(f"Core Engine Error: Picovoice process() error: {e_pv_process}")
                            # This might be a corrupted frame or an SDK issue.
                            break # Break from frame processing loop for safety
                    if self.is_listening_for_command: # If wake word was detected by the inner loop
                        pass # Handled, main 'while not self._stop_event.is_set()' loop will continue
                               # and next iteration will go into 'if self.is_listening_for_command:' block.
                else: # Should not happen if __init__ validates engine_type
                    print(f"Core Engine Error: Unknown engine_type '{self.engine_type}' in _listen loop.")
                    time.sleep(1) # Avoid spamming logs

    def _is_silent(self, data, threshold):
        """Returns 'True' if the audio chunk is below a volume threshold."""
        return audioop.rms(data, 2) < threshold

    def _process_command(self):
        """Transcribes the recorded audio and triggers the command callback."""
        print("Core Engine: Processing command...")
        full_audio_data = b"".join(self.command_audio)

        result = self.whisper_model.transcribe(full_audio_data, fp16=False)
        command_text = result["text"].strip()

        self.command_audio = []
        self.is_listening_for_command = False
        print("Core Engine: Listening for wake word...")

        if self.on_command and command_text:
            threading.Thread(target=self.on_command, args=(command_text,)).start()
