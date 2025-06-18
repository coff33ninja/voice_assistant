import os
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

# _SCRIPT_DIR should point to the project's root directory (e.g., 'e:\SCRIPTS\voice_assistant')
# This assumes config.py is in a 'modules' subdirectory.
_PROJECT_ROOT = os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir)))

BASE_DIR = os.path.join(_PROJECT_ROOT, "models")

DB_PATH = os.path.join(BASE_DIR, "reminders.db")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "fine_tuned_distilbert")

# API keys now loaded from environment variables, fallback to file if not set
def get_picovoice_key():
    return os.getenv("PICOVOICE_KEY")

def get_openweather_api_key():
    return os.getenv("OPENWEATHER_API_KEY")

PICOVOICE_KEY_FILE_PATH = os.path.join(BASE_DIR, "picovoice_key.txt")
OPENWEATHER_API_KEY_FILE_PATH = os.path.join(BASE_DIR, "openweather_api_key.txt")

ASR_DEVICE = "cuda" if hasattr(__import__('torch'), 'cuda') and __import__('torch').cuda.is_available() else "cpu"
ALIGN_LANGUAGE_CODE = "en"  # For WhisperX alignment model

GREETING_MESSAGE = "How can I help you?"

# TTS Model - Default is "tts_models/en/ljspeech/vits".
# This can be overridden by setting TTS_MODEL_NAME in the .env file (e.g., during setup).
TTS_MODEL_NAME = os.getenv("TTS_MODEL_NAME", "tts_models/en/ljspeech/vits")
TTS_SPEED_RATE = float(os.getenv("TTS_SPEED_RATE", "1.0")) # 1.0 is normal speed. <1.0 is slower, >1.0 is faster.
TTS_SAMPLERATE = 22050

# STT Model
# Default STT Model. This can be overridden by STT_MODEL_NAME in .env, or selected via whisperx_setup.py.
DEFAULT_STT_MODEL_NAME = os.getenv("STT_MODEL_NAME", "base")
STT_COMPUTE_TYPE = "int8" # Default compute type, can also be configured if needed
STT_BATCH_SIZE = 16

# Audio Recording
AUDIO_SAMPLE_RATE = 16000
AUDIO_DURATION_SECONDS = 5

# LLM
LLM_MODEL_NAME = "llama2"
