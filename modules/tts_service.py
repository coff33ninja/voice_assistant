import asyncio
import sounddevice as sd
import torch
from TTS.api import TTS as CoquiTTS
from .config import TTS_MODEL_NAME, TTS_SAMPLERATE # Use the single configured model name

tts_instance = None

def initialize_tts():
    global tts_instance
    print("Initializing TTS service...")
    
    # TTS_MODEL_NAME is loaded from .env (set during setup_tts) or defaults from config.py
    if not TTS_MODEL_NAME:
        print("Error: TTS model identifier is not configured. Cannot initialize TTS.")
        raise RuntimeError("TTS model not configured.")

    try:
        print(f"TTS service: Attempting to load model '{TTS_MODEL_NAME}'")
        tts_instance = CoquiTTS(
            model_name=TTS_MODEL_NAME,
            progress_bar=True, # Good for first use, might download/setup
            gpu=torch.cuda.is_available(),
        )
        print(f"TTS service initialized successfully with model: {TTS_MODEL_NAME}.")
    except Exception as e:
        print(f"ERROR: Failed to initialize Coqui TTS with model '{TTS_MODEL_NAME}': {e}")
        print("Please check your TTS configuration, model files, and dependencies.")
        raise # Re-raise to indicate critical failure

async def text_to_speech_async(text: str):
    if tts_instance is None:
        raise RuntimeError("TTS service not initialized. Call initialize_tts() first.")
    try:
        audio_output = await asyncio.to_thread(tts_instance.tts, text=text)
        await asyncio.to_thread(sd.play, audio_output, samplerate=TTS_SAMPLERATE)
        await asyncio.to_thread(sd.wait)
    except Exception as e:
        print(f"Async Coqui TTS error: {e}")

def text_to_speech(text: str): # Keep sync version if used by non-async parts
    if tts_instance is None:
        raise RuntimeError("TTS not initialized")
    audio = tts_instance.tts(text=text)
    sd.play(audio, samplerate=TTS_SAMPLERATE)
    sd.wait()