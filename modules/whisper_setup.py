"""
Module for using OpenAI Whisper (standard, not WhisperX) for speech-to-text.
"""
import os
import torch
import numpy as np
import tempfile
import sounddevice as sd
import scipy.io.wavfile as wav
from dotenv import set_key
from modules.config import STT_MODEL_NAME

try:
    import whisper
except ImportError:
    whisper = None

MODEL_CACHE = {}

WHISPER_MODEL_SIZES = [
    "tiny", "base", "small", "medium", "large"
]


def load_whisper_model(model_size: str = "base", device: str = "cpu"):
    """Load and cache a Whisper model of the given size."""
    global MODEL_CACHE
    if whisper is None:
        raise ImportError("The 'whisper' package is not installed. Please install it with 'pip install openai-whisper'.")
    if (model_size, device) not in MODEL_CACHE:
        MODEL_CACHE[(model_size, device)] = whisper.load_model(model_size, device=device)
    return MODEL_CACHE[(model_size, device)]


def transcribe_with_whisper(audio_np: np.ndarray, model_size: str = "base", device: str = "cpu") -> str:
    """Transcribe audio using OpenAI Whisper."""
    model = load_whisper_model(model_size, device)
    # Whisper expects float32 numpy array in [-1, 1]
    if audio_np.dtype != np.float32:
        audio_np = audio_np.astype(np.float32) / 32768.0
    result = model.transcribe(audio_np)
    return result.get("text", "")


def get_whisper_model_descriptions():
    return {
        "tiny": "Fastest, lowest accuracy, smallest size. Good for quick tests or low-resource devices.",
        "base": "Fast, small, but less accurate than larger models.",
        "small": "Balanced speed and accuracy for many use cases.",
        "medium": "Slower, more accurate, larger size.",
        "large": "Slowest, highest accuracy, largest size. Best for quality, needs most resources."
    }


def setup_whisper():
    """Interactive setup for OpenAI Whisper (standard) model selection and test."""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    dotenv_path = os.path.join(project_root, '.env')

    current_stt_model = os.getenv("STT_MODEL_NAME", STT_MODEL_NAME)
    print(f"Current Whisper model: {current_stt_model}")

    print("\nAvailable Whisper models:")
    for i, model_name in enumerate(WHISPER_MODEL_SIZES):
        print(f"{i + 1}. {model_name}")

    selected_model_name = current_stt_model
    while True:
        try:
            choice = input(f"Select a model by number (or press Enter to keep '{current_stt_model}'): ")
            if not choice:
                break
            model_index = int(choice) - 1
            if 0 <= model_index < len(WHISPER_MODEL_SIZES):
                selected_model_name = WHISPER_MODEL_SIZES[model_index]
                break
            else:
                print("Invalid selection. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    if selected_model_name != current_stt_model:
        set_key(dotenv_path, "STT_MODEL_NAME", selected_model_name)
        print(f"Whisper model set to: {selected_model_name} in {dotenv_path}")
        os.environ["STT_MODEL_NAME"] = selected_model_name
    else:
        print(f"Keeping current model: {selected_model_name}")

    try:
        print(f"\nLoading Whisper model: {selected_model_name}...")
        model = load_whisper_model(selected_model_name, device="cpu")
        print(f"Model {selected_model_name} loaded successfully.")
        print("Whisper setup complete. Models will download on first use if not already present.")
        duration = 5  # seconds
        fs = 16000
        print("Please say something after the beep...")
        sd.sleep(500)
        print("Beep!")
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
        sd.wait()
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmpfile:
            wav.write(tmpfile.name, fs, recording)
            tmp_wav_path = tmpfile.name
        # Transcribe with whisper
        result = model.transcribe(tmp_wav_path)
        print("Transcription result:", result)
        os.remove(tmp_wav_path)
    except ImportError as ie:
        print(f"A required module is missing: {ie}")
        print("Please ensure all dependencies are installed. You might need to run: pip install openai-whisper sounddevice scipy python-dotenv")
    except Exception as e:
        print(f"Whisper setup or test failed: {e}")
