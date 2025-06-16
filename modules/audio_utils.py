import asyncio
import numpy as np
import sounddevice as sd
from .config import AUDIO_SAMPLE_RATE, AUDIO_DURATION_SECONDS

async def record_audio_async(sample_rate=AUDIO_SAMPLE_RATE, duration=AUDIO_DURATION_SECONDS) -> np.ndarray:
    print("Recording (async)...")
    recording_data = await asyncio.to_thread(
        sd.rec, int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="int16"
    )
    await asyncio.to_thread(sd.wait)
    print("Recording complete (async).")
    return recording_data.flatten()

def record_audio(sample_rate=AUDIO_SAMPLE_RATE, duration=AUDIO_DURATION_SECONDS) -> np.ndarray:
    print("Recording...")
    recording = sd.rec(
        int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="int16"
    )
    sd.wait()
    print("Recording complete.")
    return recording.flatten()