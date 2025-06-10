# voice_assistant/core/engine.py

import openwakeword
import pyaudio
import whisper
import audioop
import threading
import time
from collections import deque


class VoiceCore:
    def __init__(
        self,
        wake_word_model_path,
        on_wake_word=None,
        on_command=None,
        whisper_model_name="base.en",
        silence_threshold=500,
    ):
        """
        Initializes the Voice Core.

        Args:
            wake_word_model_path (str): The path to the .onnx wake word model file.
            on_wake_word (function): A callback function to execute when the wake word is detected.
            on_command (function): A callback function to execute with the transcribed command text.
        """
        self.wake_word_model_path = wake_word_model_path
        self.on_wake_word = on_wake_word
        self.on_command = on_command
        self.silence_threshold = silence_threshold

        # --- Initialize Audio Stream (PyAudio) ---
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=1280,
        )

        # --- Initialize Wake Word Engine (openWakeWord) ---
        self.oww = openwakeword.Model(wakeword_models=[self.wake_word_model_path])

        # --- Initialize Speech-to-Text Engine (Whisper) ---
        self.whisper_model = whisper.load_model("base.en")
        self.model_name = wake_word_model_path.split("/")[-1].replace(".onnx", "")

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
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
        print("Core Engine: Stopped gracefully.")

    def _listen(self):
        """The main loop that processes audio chunks."""
        while not self._stop_event.is_set():
            try:
                audio_chunk = self.stream.read(1280, exception_on_overflow=False)
            except IOError as e:
                # This can happen if the stream is closed while reading
                if e.errno == pyaudio.paInputOverflowed:
                    print("Core Engine Warning: Input overflowed. Dropping frame.")
                    continue
                break

            if self.is_listening_for_command:
                self.command_audio.append(audio_chunk)
                if self._is_silent(audio_chunk, threshold=self.silence_threshold):
                    self._process_command()
            else:
                self.audio_buffer.append(audio_chunk)
                prediction = self.oww.predict(audio_chunk)
                if prediction[self.model_name] > 0.5:  # Use the dynamic model name
                    self.is_listening_for_command = True
                    self.command_audio = list(self.audio_buffer)
                    if self.on_wake_word:
                        threading.Thread(target=self.on_wake_word).start()

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
