import os
import json
from modules.install_dependencies import install_dependencies
from modules.download_and_models import setup_tts, setup_precise
from modules.api_key_setup import setup_api_key
from modules.whisperx_setup import setup_whisperx
from modules.db_setup import setup_db
from modules.utils import create_directories
from modules.config import (
    DB_PATH,
    PICOVOICE_KEY_FILE_PATH,
    OPENWEATHER_API_KEY_FILE_PATH,
)

# Determine the absolute path to the 'models' directory relative to this script
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(_SCRIPT_DIR, "models")

PRECISE_MODEL_URL = "https://github.com/MycroftAI/mycroft-precise/releases/download/v0.3.0/precise-engine_0.3.0_x86_64.tar.gz"
DATASET_PATH = os.path.join(BASE_DIR, "intent_dataset.csv")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "fine_tuned_distilbert")
SETUP_CHECKPOINTS_PATH = os.path.join(BASE_DIR, "setup_checkpoints.json")

SETUP_STEPS = [
    "dependencies",
    "tts",
    "precise",
    "picovoice_api_key",
    "whisperx",
    "db",
    "dataset",
    "model_training",
    "openweather_api_key"
    ]

def load_checkpoints():
    if os.path.exists(SETUP_CHECKPOINTS_PATH):
        with open(SETUP_CHECKPOINTS_PATH, "r") as f:
            return json.load(f)
    return {step: False for step in SETUP_STEPS}

def save_checkpoints(checkpoints):
    with open(SETUP_CHECKPOINTS_PATH, "w") as f:
        json.dump(checkpoints, f, indent=2)

def main():
    print("Setting up voice assistant...")
    create_directories(BASE_DIR, MODEL_SAVE_PATH)
    checkpoints = load_checkpoints()
    if not checkpoints.get("dependencies", False):
        install_dependencies()
        checkpoints["dependencies"] = True
        save_checkpoints(checkpoints)
    # Always run TTS setup to allow voice model changes
    setup_tts()
    checkpoints["tts"] = True # Still mark as 'done' for consistency, though it's always run
    save_checkpoints(checkpoints)
    if not checkpoints.get("precise", False):
        setup_precise(BASE_DIR, PRECISE_MODEL_URL)
        checkpoints["precise"] = True
        save_checkpoints(checkpoints)
    if not checkpoints.get("picovoice_api_key", False):
        setup_api_key(PICOVOICE_KEY_FILE_PATH, "Picovoice", "Enter Picovoice Access Key (or press Enter to skip): ")
        checkpoints["picovoice_api_key"] = True
        save_checkpoints(checkpoints)
    if not checkpoints.get("openweather_api_key", False):
        setup_api_key(OPENWEATHER_API_KEY_FILE_PATH, "OpenWeather", "Enter OpenWeather API Key (or press Enter to skip): ")
        checkpoints["openweather_api_key"] = True
        save_checkpoints(checkpoints)
    if not checkpoints.get("whisperx", False):
        setup_whisperx()
        checkpoints["whisperx"] = True
        save_checkpoints(checkpoints)
    if not checkpoints.get("db", False):
        setup_db(DB_PATH)
        checkpoints["db"] = True
        save_checkpoints(checkpoints)
    from modules.dataset import create_dataset
    if not checkpoints.get("dataset", False):
        create_dataset(DATASET_PATH)
        checkpoints["dataset"] = True
        save_checkpoints(checkpoints)
    from modules.model_training import fine_tune_model
    if not checkpoints.get("model_training", False):
        fine_tune_model(DATASET_PATH, MODEL_SAVE_PATH)
        checkpoints["model_training"] = True
        save_checkpoints(checkpoints)
    print("Setup complete. Run voice_assistant.py to start the assistant.")

if __name__ == "__main__":
    main()
