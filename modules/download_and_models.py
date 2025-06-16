import os
import urllib.request
import subprocess
from TTS.api import TTS as CoquiTTS # Renamed to avoid conflict if TTS is a common var name

def download_file(url, dest):
    try:
        urllib.request.urlretrieve(url, dest)
        print(f"Downloaded {dest}")
    except Exception as e:
        print(f"Failed to download {url}: {e}")

def setup_tts(base_dir):
    # base_dir is not directly used by CoquiTTS for model storage, it uses a cache.
    # The ImportError related to ModelManager is bypassed by directly using TTS.api.TTS
    try:
        print("Initializing Coqui TTS to download model (if needed)...")
        # This will download "tts_models/en/ljspeech/vits" to the default TTS cache path
        # if it's not already there. This is the model used in voice_assistant.py.
        CoquiTTS(model_name="tts_models/en/ljspeech/vits", progress_bar=True)
        print("Coqui VITS model 'tts_models/en/ljspeech/vits' is ready.")
    except Exception as e:
        print(f"Failed to initialize/download Coqui TTS model: {e}")
        print("Please ensure you have a working internet connection and TTS dependencies are correctly installed.")


def setup_precise(base_dir, model_url):
    precise_model = os.path.join(base_dir, "precise-engine.tar.gz")
    precise_dir = os.path.join(base_dir, "precise-engine")
    if not os.path.exists(precise_dir):
        download_file(model_url, precise_model)
        subprocess.run(["tar", "-xzf", precise_model, "-C", base_dir], check=True)
    print(
        "To train a custom wakeword, record 10-20 samples and use precise-train: https://github.com/MycroftAI/mycroft-precise"
    )
