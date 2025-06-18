import asyncio
import sounddevice as sd
import torch
import os
from TTS.api import TTS as CoquiTTS
from .config import (
    TTS_MODEL_NAME,
    TTS_SPEED_RATE,
    TTS_SAMPLERATE,
    DEFAULT_SPEAKER_WAV_PATH as DEFAULT_SPEAKER_WAV_PATH_TTS_SERVICE, # Use centralized path
    TTS_DEVICE  # Import TTS_DEVICE
)  # Use the single configured model name

tts_instance = None


def initialize_tts():
    global tts_instance
    print(f"TTS Service: Initializing with configured device: {TTS_DEVICE}") # Added logging

    # TTS_MODEL_NAME is loaded from .env (set during setup_tts) or defaults from config.py
    if not TTS_MODEL_NAME:
        print("Error: TTS model identifier is not configured. Cannot initialize TTS.")
        raise RuntimeError("TTS model not configured.")

    try:
        print(f"TTS service: Attempting to load model '{TTS_MODEL_NAME}'")

        use_gpu = False
        if TTS_DEVICE.lower() == "cuda":
            if torch.cuda.is_available():
                use_gpu = True
                print("TTS Service: CUDA device selected and torch.cuda.is_available(). Will attempt to use GPU.")
            else:
                print("TTS Service: CUDA device selected, but torch.cuda.is_available() is False. Falling back to CPU.")
        elif TTS_DEVICE.lower() == "cpu":
            print("TTS Service: CPU device selected.")
        else:
            print(f"TTS Service: Unknown TTS_DEVICE '{TTS_DEVICE}'. Defaulting to CPU.")

        tts_instance = CoquiTTS(
            model_name=TTS_MODEL_NAME,
            progress_bar=False,  # Setup script handles initial download with progress bar
            gpu=use_gpu,
        )

        if use_gpu and tts_instance.device.type == 'cuda':
            print(f"TTS service initialized successfully with model: {TTS_MODEL_NAME} on GPU.")
        elif not use_gpu and tts_instance.device.type == 'cpu':
            print(f"TTS service initialized successfully with model: {TTS_MODEL_NAME} on CPU.")
        else:
            # This case might occur if CoquiTTS falls back to CPU despite gpu=True request,
            # or if there's a mismatch in device checking.
            print(f"TTS service initialized successfully with model: {TTS_MODEL_NAME}. Note: Actual device is {tts_instance.device.type}, requested use_gpu was {use_gpu}.")

    except Exception as e:
        print(
            f"ERROR: Failed to initialize Coqui TTS with model '{TTS_MODEL_NAME}': {e}"
        )
        print("Please check your TTS configuration, model files, and dependencies.")
        raise  # Re-raise to indicate critical failure


async def text_to_speech_async(text: str):
    if tts_instance is None:
        raise RuntimeError("TTS service not initialized. Call initialize_tts() first.")
    try:
        tts_call_kwargs = {"text": text, "speed": TTS_SPEED_RATE}
        playback_samplerate = TTS_SAMPLERATE # Default samplerate

        # Check if the globally configured TTS_MODEL_NAME is an XTTS model
        if "xtts" in TTS_MODEL_NAME.lower():
            print(f"TTS Service: XTTS model '{TTS_MODEL_NAME}' detected.")
            tts_call_kwargs["language"] = "en" # Default language for XTTS
            if os.path.exists(DEFAULT_SPEAKER_WAV_PATH_TTS_SERVICE):
                tts_call_kwargs["speaker_wav"] = DEFAULT_SPEAKER_WAV_PATH_TTS_SERVICE
                print(f"TTS Service: Using speaker_wav: {DEFAULT_SPEAKER_WAV_PATH_TTS_SERVICE}")
            else:
                print(f"WARNING (TTS Service): Default speaker WAV for XTTS models not found at '{DEFAULT_SPEAKER_WAV_PATH_TTS_SERVICE}'. XTTS may fail or use a default voice.")

        audio_output = await asyncio.to_thread(tts_instance.tts, **tts_call_kwargs)

        # Determine the correct samplerate for playback, especially for XTTS
        if hasattr(tts_instance, 'synthesizer') and tts_instance.synthesizer is not None and hasattr(tts_instance.synthesizer, 'output_sample_rate'):
            playback_samplerate = tts_instance.synthesizer.output_sample_rate
            print(f"TTS Service: Using model's output samplerate: {playback_samplerate}")

        await asyncio.to_thread(sd.play, audio_output, samplerate=playback_samplerate)
        await asyncio.to_thread(sd.wait)
    except Exception as e:
        print(f"Async Coqui TTS error: {e}")


def text_to_speech(text: str): # Sync version
    if tts_instance is None:
        raise RuntimeError("TTS not initialized")

    # Replicate XTTS logic for sync version (simplified, assumes tts_instance is already XTTS if configured)
    tts_call_kwargs = {"text": text, "speed": TTS_SPEED_RATE}
    if "xtts" in TTS_MODEL_NAME.lower(): # Use configured model name
        tts_call_kwargs["language"] = "en"
        if os.path.exists(DEFAULT_SPEAKER_WAV_PATH_TTS_SERVICE): # Check speaker wav
            tts_call_kwargs["speaker_wav"] = DEFAULT_SPEAKER_WAV_PATH_TTS_SERVICE

    audio = tts_instance.tts(**tts_call_kwargs)
    # Samplerate for sync version - ideally also checks model's output_sample_rate
    playback_samplerate_sync = TTS_SAMPLERATE
    if hasattr(tts_instance, 'synthesizer') and tts_instance.synthesizer is not None and hasattr(tts_instance.synthesizer, 'output_sample_rate'):
        playback_samplerate_sync = tts_instance.synthesizer.output_sample_rate
    sd.play(audio, samplerate=playback_samplerate_sync)
    sd.wait()
