import asyncio
import sounddevice as sd
import torch
from TTS.api import TTS as CoquiTTS
from .config import TTS_MODEL_NAME, TTS_SAMPLERATE

tts_instance = None

def initialize_tts():
    global tts_instance
    print("Initializing TTS service...")
    try:
        tts_instance = CoquiTTS(
            model_name=TTS_MODEL_NAME,
            progress_bar=True,
            gpu=torch.cuda.is_available(),
        )
        print("TTS service initialized.")
    except Exception as e:
        print(f"Failed to initialize Coqui TTS: {e}")
        raise

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