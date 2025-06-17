import os
import sys
import urllib.request
import subprocess
from TTS.api import TTS as CoquiTTS # Renamed to avoid conflict if TTS is a common var name
from dotenv import set_key # For .env manipulation
from .config import _PROJECT_ROOT, TTS_MODEL_NAME as CURRENT_EFFECTIVE_TTS_MODEL

def download_file(url, dest):
    try:
        urllib.request.urlretrieve(url, dest)
        print(f"Downloaded {dest}")
    except Exception as e:
        print(f"Failed to download {url}: {e}")

def setup_tts():
    env_path = os.path.join(_PROJECT_ROOT, ".env")

    print("\n--- TTS Model Configuration ---")
    print("You can choose a TTS model from the Coqui TTS model zoo.")
    print("Attempting to list available Coqui TTS models...")
    try:
        # Execute the command to list models
        result = subprocess.run(
            [sys.executable, "-m", "TTS.server.server", "--list_models"],
            capture_output=True, text=True, check=False
        )
        if result.returncode == 0 and result.stdout:
            print("Available Coqui TTS models:\n")
            print(result.stdout)
        else:
            print("Could not automatically list models. Please ensure Coqui TTS is installed correctly.")
            print("You can try running 'python -m TTS.server.server --list_models' manually in your terminal.")
    except Exception as e:
        print(f"Error trying to list TTS models: {e}")
        print("Please ensure Coqui TTS is installed and accessible.")
    print("\nExample model name: 'tts_models/en/ljspeech/vits' or 'tts_models/en/vctk/vits_base'")

    # CURRENT_EFFECTIVE_TTS_MODEL is what the application would currently use (from .env or config's default)
    print(f"\nCurrent effective TTS model: {CURRENT_EFFECTIVE_TTS_MODEL}")
    user_input_model_name = input(f"Enter new TTS model name (or press Enter to keep '{CURRENT_EFFECTIVE_TTS_MODEL}'): ").strip()

    final_model_to_use = CURRENT_EFFECTIVE_TTS_MODEL # Start with the current effective model

    if user_input_model_name: # User entered a new model name
        final_model_to_use = user_input_model_name
        # Attempt to save the new model name to .env
        # set_key will create .env if it doesn't exist.
        if set_key(env_path, "TTS_MODEL_NAME", final_model_to_use, quote_mode="always"):
            print(f"TTS model name set to '{final_model_to_use}' in {env_path}")
        else:
            # This case should be rare if directory is writable.
            print(f"Warning: Could not save TTS_MODEL_NAME to {env_path}. Using '{final_model_to_use}' for this session.")
            print(f"Please ensure the directory {_PROJECT_ROOT} is writable or manually set TTS_MODEL_NAME='{final_model_to_use}' in {env_path}")
    else:
        print(f"Keeping current TTS model: {final_model_to_use}")

    try:
        print(f"Initializing Coqui TTS with model '{final_model_to_use}' to download (if needed)...")
        CoquiTTS(model_name=final_model_to_use, progress_bar=True)
        print(f"Coqui TTS model '{final_model_to_use}' is ready.")
    except Exception as e:
        print(f"ERROR: Failed to initialize/download Coqui TTS model '{final_model_to_use}': {e}")
        print("Please ensure:")
        print("  - You have a working internet connection.")
        print("  - The model name is correct and available in the Coqui TTS zoo (check with `python -m TTS.server.server --list_models`).")
        print("  - TTS dependencies are correctly installed.")
        raise # Re-raise to indicate setup step failure

def setup_precise(base_dir, model_url):
    precise_model = os.path.join(base_dir, "precise-engine.tar.gz")
    precise_dir = os.path.join(base_dir, "precise-engine")
    if not os.path.exists(precise_dir):
        download_file(model_url, precise_model)
        subprocess.run(["tar", "-xzf", precise_model, "-C", base_dir], check=True)
    print(
        "To train a custom wakeword, record 10-20 samples and use precise-train: https://github.com/MycroftAI/mycroft-precise"
    )
