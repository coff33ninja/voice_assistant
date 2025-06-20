"""
Module for using OpenAI Whisper (standard, not WhisperX) for speech-to-text.
"""
import os
import torch
import numpy as np

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
