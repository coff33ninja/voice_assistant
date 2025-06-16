import os
import torch

# _SCRIPT_DIR should point to the project's root directory (e.g., 'e:\SCRIPTS\voice_assistant')
# This assumes config.py is in a 'modules' subdirectory.
_PROJECT_ROOT = os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir)))

BASE_DIR = os.path.join(_PROJECT_ROOT, "models")

DB_PATH = os.path.join(BASE_DIR, "reminders.db")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "fine_tuned_distilbert")
PICOVOICE_KEY_FILE_PATH = os.path.join(
    BASE_DIR, "picovoice_key.txt"
)  # Renamed for clarity
OPENWEATHER_API_KEY_FILE_PATH = os.path.join(BASE_DIR, "openweather_api_key.txt")

ASR_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ALIGN_LANGUAGE_CODE = "en"  # For WhisperX alignment model

GREETING_MESSAGE = "How can I help you?"

# TTS Model
TTS_MODEL_NAME = "tts_models/en/ljspeech/vits"
TTS_SAMPLERATE = 22050

# STT Model
STT_MODEL_NAME = "base.en" # or "base" if multilingual needed and handled
STT_COMPUTE_TYPE = "int8"
STT_BATCH_SIZE = 16

# Audio Recording
AUDIO_SAMPLE_RATE = 16000
AUDIO_DURATION_SECONDS = 5

# LLM
LLM_MODEL_NAME = "llama2"
