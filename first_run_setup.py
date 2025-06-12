"""
first_run_setup.py

Guides the user through initial configuration of the voice assistant, including:
- Picovoice/OpenWakeWord selection
- Wake word selection
- TTS engine and voice selection
- Configuration saving
"""

import os
import sys
import time
import re
import numpy as np
import urllib.error # Added for specific network error handling
import urllib.request
import hashlib
import logging
import glob
from dotenv import load_dotenv, set_key
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    filename="setup.log",
    format="%(asctime)s - %(levelname)s - %(message)s",
)

load_dotenv()

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = script_dir
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

try:
    from core.tts import speak, tts_engine
    from core.user_config import load_config, save_config, DEFAULT_CONFIG
except ImportError as e:
    logging.error(f"Could not import core modules: {e}")
    print(f"Error: Could not import core modules: {e}")
    sys.exit(1)

AVAILABLE_WAKE_WORDS = [
    {"name": "Computer", "model_file": "wakeword_models/Computer.ppn"},
    {"name": "Jarvis", "model_file": "wakeword_models/Jarvis.ppn"},
    {"name": "Assistant", "model_file": "wakeword_models/Assistant.ppn"},
]

OPENWAKEWORD_MODELS = [
    {
        "name": "Computer",
        "url": "https://github.com/synesthesiam/openwakeword-models/raw/main/onnx/computer.onnx",
        "model_file": "wakeword_models/Computer.onnx",
        "checksum": "SHA256_PLACEHOLDER_COMPUTER", # Placeholder: Replace with actual SHA256 hash
    },
    {
        "name": "Jarvis",
        "url": "https://github.com/synesthesiam/openwakeword-models/raw/main/onnx/jarvis.onnx",
        "model_file": "wakeword_models/Jarvis.onnx",
        "checksum": "SHA256_PLACEHOLDER_JARVIS", # Placeholder: Replace with actual SHA256 hash
    },
    {
        "name": "Assistant",
        "url": "https://github.com/synesthesiam/openwakeword-models/raw/main/onnx/assistant.onnx",
        "model_file": "wakeword_models/Assistant.onnx",
        "checksum": "SHA256_PLACEHOLDER_ASSISTANT", # Placeholder: Replace with actual SHA256 hash
    },
]

AVAILABLE_TTS_ENGINES = [
    {"name": "pyttsx3", "id": "pyttsx3", "description": "Offline TTS (pyttsx3)"},
    {"name": "gTTS", "id": "gtts", "description": "Google Text-to-Speech (online)"},
]


def verify_checksum(file_path, expected_checksum):
    try:
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)
        return sha256.hexdigest() == expected_checksum
    except Exception as e:
        logging.error(f"Checksum verification failed for {file_path}: {e}")
        return False


def download_with_progress(url, path, retries=3, delay=5):
    """Downloads a file with progress and retries on common network errors."""
    for attempt in range(retries):
        try:
            desc = os.path.basename(path)
            if retries > 1:
                desc += f" (Attempt {attempt + 1}/{retries})"
            with tqdm(unit="B", unit_scale=True, desc=desc) as pbar:
                def report(blocknum, blocksize, totalsize):
                    pbar.total = totalsize # Set total size dynamically
                    pbar.update(blocksize) # Increment by blocksize

                urllib.request.urlretrieve(url, path, reporthook=report)
            return  # Success
        except (urllib.error.URLError, ConnectionResetError, TimeoutError) as e:
            logging.warning(f"Download attempt {attempt + 1} for {url} failed: {e}")
            if attempt < retries - 1:
                print(f"Download failed: {e}. Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                logging.error(f"Failed to download {url} after {retries} attempts.")
                raise RuntimeError(f"Download failed after {retries} attempts: {e}")
        except Exception as e:  # Catch other unexpected errors during download
            logging.error(f"Unexpected error during download of {url} (attempt {attempt + 1}): {e}")
            raise RuntimeError(f"Unexpected download error: {e}")


def download_openwakeword_models():
    if not os.path.exists("wakeword_models"):
        os.makedirs("wakeword_models")
        logging.info("Created wakeword_models directory")

    local_models = [
        {"name": os.path.basename(f).replace(".onnx", ""), "model_file": f}
        for f in glob.glob("wakeword_models/*.onnx")
    ]
    for model in local_models:
        if model["name"] not in [m["name"] for m in OPENWAKEWORD_MODELS]:
            OPENWAKEWORD_MODELS.append(model)
            logging.info(f"Added local model: {model['name']}")

    for model in OPENWAKEWORD_MODELS:
        path = model["model_file"]
        if not os.path.exists(path):
            if "url" not in model:
                print(
                    f"No URL provided for {model['name']}. Please place the .onnx file at {path}."
                )
                logging.warning(f"No URL for model {model['name']}")
                continue
            logging.info(
                f"Downloading OpenWakeWord model for {model['name']} from {model['url']}"
            )
            print(f"Downloading OpenWakeWord model for {model['name']}...")
            try:
                download_with_progress(model["url"], path) # This now has retries
                download_successful = True
            except RuntimeError as e_download: # Catch from download_with_progress if all retries fail
                print(f"ERROR: Unable to download {model['name']} from {model['url']} after multiple attempts: {e_download}")
                logging.error(f"Final download failed for {model['url']}: {e_download}")
                speak(f"Failed to download the model for {model['name']}. You may need to download it manually or check your connection.")
                download_successful = False
            except Exception as e_unexpected_download: # Broader catch
                print(f"ERROR: An unexpected error occurred while trying to download {model['url']}: {e_unexpected_download}")
                logging.error(f"Unexpected download error for {model['url']}: {e_unexpected_download}")
                download_successful = False

            if download_successful and os.path.exists(path):
                print(f"Successfully downloaded: {path}")
                # Checksum verification only if a non-placeholder checksum is provided
                expected_checksum = model.get("checksum")
                is_placeholder_checksum = isinstance(expected_checksum, str) and expected_checksum.startswith("SHA256_PLACEHOLDER_")

                if expected_checksum and not is_placeholder_checksum:
                    print(f"Verifying checksum for {model['name']}...")
                    if not verify_checksum(path, expected_checksum):
                        print(f"ERROR: Checksum mismatch for {path}. The file might be corrupted or the expected checksum is incorrect.")
                        logging.error(f"Checksum mismatch for {path}. Expected: {expected_checksum}, Got: <actual_hash_not_logged_here_for_brevity>")
                        speak(f"The downloaded file for {model['name']} failed a security check. It will be removed.")
                        try:
                            os.remove(path)
                            print(f"Removed potentially corrupted file: {path}")
                        except OSError as e_del:
                            print(f"ERROR: Could not remove file {path} after checksum failure: {e_del}")
                            logging.error(f"Failed to remove {path} after checksum error: {e_del}")
                    else:
                        print(f"Checksum verified for {model['name']}.")
                        logging.info(f"Successfully downloaded and verified {path}")
                elif is_placeholder_checksum:
                    print(f"INFO: Checksum for {model['name']} is a placeholder. Skipping verification. For enhanced security, please update the script with the actual SHA256 hash for this model.")
                    logging.warning(f"Skipping checksum verification for {path} due to placeholder. Downloaded without verification.")
                else: # No checksum provided or it's empty/None
                    print(f"INFO: No checksum provided for {model['name']}. Skipping verification.")
                    logging.info(f"Successfully downloaded {path} (checksum not provided/verified).")
        else:
            print(f"Model already exists: {path}")
            logging.info(f"Model found: {path}")

def transcribe_audio_with_whisper(whisper_model, audio_bytes) -> str:
    try:
        audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
        audio_float32 = audio_int16.astype(np.float32) / 32768.0
        result = whisper_model.transcribe(audio_float32, fp16=False)
        transcribed_text = result["text"]
        if isinstance(transcribed_text, list):
            transcribed_text = " ".join(str(x) for x in transcribed_text)
        return transcribed_text.strip().lower() if transcribed_text else ""
    except Exception as e:
        logging.error(f"Whisper transcription error: {e}")
        print(f"Error during Whisper transcription: {e}")
        return ""


def match_choice_from_text(
    transcribed_text: str, options: list[dict], key: str = "name"
) -> int:
    if not transcribed_text:
        return -1
    text = transcribed_text.strip().lower()
    for i, opt in enumerate(options):
        if opt[key].lower() in text:
            return i
        if re.search(
            r"(number|option|#)?\s*" + str(i + 1) + r"\b", text, re.IGNORECASE
        ) or text == str(i + 1):
            return i
    number_words = [
        "zero",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
    ]
    for i, opt in enumerate(options):
        if i < len(number_words) and number_words[i] in text:
            return i
    return -1


def select_wake_word_engine():
    print("\nChoose your wake word engine:")
    print(
        "1. Picovoice Porcupine (.ppn, requires Access Key, high accuracy, closed-source)"
    )
    print(
        "2. OpenWakeWord (.onnx, open-source, no key required, supports whispered wake words)"
    )
    logging.info("Prompting user to select wake word engine")
    while True:
        choice = input("Enter 1 for Picovoice or 2 for OpenWakeWord: ").strip()
        if choice == "1":
            logging.info("User selected Picovoice")
            return "picovoice"
        elif choice == "2":
            logging.info("User selected OpenWakeWord")
            return "openwakeword"
        else:
            print("Invalid input. Please enter 1 or 2.")
            logging.warning("Invalid engine selection input")


def select_tts_engine():
    print("\nChoose your TTS engine:")
    for i, engine in enumerate(AVAILABLE_TTS_ENGINES):
        print(f"{i + 1}. {engine['name']} ({engine['description']})")
        speak(f"Option {i + 1}: {engine['name']}")
        time.sleep(0.3)
    logging.info("Prompting user to select TTS engine")
    while True:
        choice = input("Enter the number for your chosen TTS engine: ").strip()
        try:
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(AVAILABLE_TTS_ENGINES):
                logging.info(
                    f"User selected TTS engine: {AVAILABLE_TTS_ENGINES[choice_idx]['name']}"
                )
                return AVAILABLE_TTS_ENGINES[choice_idx]["id"]
            else:
                print(
                    f"Invalid number. Please choose between 1 and {len(AVAILABLE_TTS_ENGINES)}."
                )
        except ValueError:
            print("Please enter a valid number.")
        logging.warning("Invalid TTS engine selection input")


def _get_choice_via_voice_input(
    prompt_context: str,
    options: list[dict],
    key: str = "name",
    attempts: int = 3,
    listen_duration_sec: int = 3,
) -> int:
    chosen_index = -1
    temp_voice_core = None
    try:
        logging.info(f"Initializing VoiceCore for STT ({prompt_context} selection)")
        print(
            f"INFO: Initializing temporary VoiceCore for STT ({prompt_context} selection)..."
        )
        from core.engine import VoiceCore

        temp_voice_core = VoiceCore(
            engine_type="openwakeword", openwakeword_model_path=None
        )
        for attempt_num in range(attempts):
            speak(
                f"{prompt_context} Choice Attempt {attempt_num + 1}. Please speak now."
            )
            print(
                f"Listening for {prompt_context.lower()} choice ({listen_duration_sec} seconds)..."
            )
            logging.info(
                f"Listening for {prompt_context} choice, attempt {attempt_num + 1}"
            )
            time.sleep(0.5)
            recorded_audio_frames = []
            transcribed_text = ""
            if (
                temp_voice_core
                and hasattr(temp_voice_core, "stream")
                and temp_voice_core.stream
                and not temp_voice_core.stream.is_stopped()
            ):
                for _ in range(int(16000 / 1280 * listen_duration_sec)):
                    try:
                        audio_chunk = temp_voice_core.stream.read(
                            1280, exception_on_overflow=False
                        )
                        recorded_audio_frames.append(audio_chunk)
                    except IOError as e_read:
                        logging.error(
                            f"Audio stream read error during {prompt_context} selection: {e_read}"
                        )
                        print(
                            f"Error reading audio stream during {prompt_context} selection: {e_read}"
                        )
                        break
                logging.info(f"{prompt_context} choice recording finished")
                print(f"{prompt_context} choice recording finished.")
                if recorded_audio_frames:
                    full_audio_data = b"".join(recorded_audio_frames)
                    if (
                        hasattr(temp_voice_core, "whisper_model")
                        and temp_voice_core.whisper_model
                    ):
                        logging.info(f"Transcribing {prompt_context.lower()} choice")
                        print(f"Transcribing {prompt_context.lower()} choice...")
                        transcribed_text = transcribe_audio_with_whisper(
                            temp_voice_core.whisper_model, full_audio_data
                        )
                        if transcribed_text:
                            logging.info(
                                f"Transcribed {prompt_context.lower()} choice: '{transcribed_text}'"
                            )
                            print(
                                f"Whisper transcribed {prompt_context.lower()} choice as: '{transcribed_text}'"
                            )
            if transcribed_text:
                chosen_index = match_choice_from_text(
                    transcribed_text, options, key=key
                )
                if chosen_index != -1:
                    speak(
                        f"Okay, I understood your {prompt_context.lower()} choice by voice!"
                    )
                    logging.info(
                        f"Voice choice for {prompt_context} successful: index {chosen_index}"
                    )
                    break
    finally:
        if temp_voice_core:
            logging.info(f"Stopping VoiceCore for STT ({prompt_context} selection)")
            print(
                f"INFO: Stopping temporary VoiceCore for STT ({prompt_context} selection)..."
            )
            temp_voice_core.stop()
    return chosen_index


def _select_and_configure_tts_voice(tts_engine_type: str):
    logging.info("Starting TTS voice selection")
    speak("Now, let's choose a voice for the TTS engine.")
    time.sleep(0.5)
    available_tts_voices = []
    try:
        from core.tts import TTSEngineFactory

        temp_engine = TTSEngineFactory.create_engine(tts_engine_type)
        available_tts_voices = temp_engine.get_available_voices()
        temp_engine.stop()
    except Exception as e:
        logging.error(f"Failed to get voices for {tts_engine_type}: {e}")
        speak("I couldn't find any voices for the selected TTS engine. Using default.")
        return
    if not available_tts_voices:
        speak("No specific voices found. We'll use the default.")
        logging.warning(f"No voices found for {tts_engine_type}")
        return
    speak("Here are the available voices:")
    print("\nAvailable TTS Voices:")
    for i, voice in enumerate(available_tts_voices):
        voice_display_name = voice.get("name", f"Voice {i+1}")
        print(f"{i + 1}. {voice_display_name}")
        if i < 5:
            speak(f"Option {i + 1}: {voice_display_name}")
            time.sleep(0.2)
    if len(available_tts_voices) > 5:
        speak("More voices are listed in the console.")
    chosen_voice_id = None
    chosen_voice_obj_idx = -1
    speak("You can say the name or number of the voice you'd like.")
    time.sleep(0.5)
    chosen_voice_obj_idx = _get_choice_via_voice_input(
        prompt_context="TTS Voice", options=available_tts_voices, key="name"
    )
    if chosen_voice_obj_idx == -1:
        speak("Trouble understanding your voice choice. Let's try typing.")
        logging.info("Falling back to typed input for TTS voice")
        for attempt in range(3):
            speak("Please type the number of your chosen voice.")
            try:
                user_input = input("Enter the number for your chosen TTS voice: ")
                choice = int(user_input)
                if 1 <= choice <= len(available_tts_voices):
                    chosen_voice_obj_idx = choice - 1
                    break
                else:
                    speak("Invalid number for voice choice.")
                    logging.warning("Invalid TTS voice number input")
            except ValueError:
                speak("That wasn't a number. Try again.")
                logging.warning("Non-numeric TTS voice input")
            if attempt == 2:
                speak("Skipping voice selection.")
                logging.info("Skipped TTS voice selection")
    if chosen_voice_obj_idx != -1:
        chosen_voice_id = available_tts_voices[chosen_voice_obj_idx]["id"]
        chosen_voice_name = available_tts_voices[chosen_voice_obj_idx]["name"]
        speak(f"You've selected {chosen_voice_name} as your voice.")
        print(f"Selected TTS Voice: {chosen_voice_name} (ID: {chosen_voice_id})")
        logging.info(f"Selected TTS voice: {chosen_voice_name} (ID: {chosen_voice_id})")
        current_config = load_config()
        current_config["chosen_tts_voice_id"] = chosen_voice_id
        current_config["tts_engine_type"] = tts_engine_type
        if save_config(current_config):
            print("TTS configuration saved.")
            logging.info("TTS configuration saved")
            if tts_engine.set_voice(chosen_voice_id):
                speak(f"Now using the {chosen_voice_name} voice.")
            else:
                speak(f"Will use {chosen_voice_name} next time you start.")
                logging.warning("TTS voice set failed")
        else:
            speak("Error saving your voice choice.")
            logging.error("Failed to save TTS configuration")
    else:
        speak("Using default voice.")
        logging.info("Using default TTS voice")


def _setup_picovoice_engine() -> bool:
    try:
        # Ensure wakeword_models directory exists
        wakeword_models_dir = "wakeword_models"
        if not os.path.exists(wakeword_models_dir):
            os.makedirs(wakeword_models_dir)
            logging.info(f"Created directory: {wakeword_models_dir}")
            print(
                f"INFO: Created directory for wake word models: {os.path.abspath(wakeword_models_dir)}"
            )
        logging.info("Starting Picovoice setup")
        speak(
            "You selected Picovoice Porcupine. Let's set up your Access Key and wake word model."
        )
        print("\n[Picovoice Setup]")
        print("- You need a Picovoice Access Key from https://console.picovoice.ai/.")
        print("- Place your .ppn model files in the 'wakeword_models/' directory.")
        picovoice_access_key = os.environ.get("PICOVOICE_ACCESS_KEY")
        env_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
        if not picovoice_access_key:
            speak("No Picovoice Access Key found.")
            print(
                "No PICOVOICE_ACCESS_KEY found in environment variables or .env file."
            )
            logging.warning("No Picovoice Access Key found")
            speak("Please enter your Picovoice Access Key from the Picovoice Console.")
            for attempt in range(3):
                access_key = input("Enter your Picovoice Access Key: ").strip()
                if access_key:
                    try:
                        set_key(env_file, "PICOVOICE_ACCESS_KEY", access_key)
                        os.environ["PICOVOICE_ACCESS_KEY"] = access_key
                        print("Access Key saved to .env file.")
                        speak("Access Key saved successfully.")
                        logging.info("Picovoice Access Key saved to .env")
                        picovoice_access_key = access_key
                        break
                    except Exception as e:
                        print(f"Failed to save Access Key to .env file: {e}")
                        speak("Error saving Access Key. Try again.")
                        logging.error(f"Failed to save Picovoice Access Key: {e}")
                else:
                    speak("Access Key cannot be empty. Try again.")
                    logging.warning("Empty Picovoice Access Key input")
                if attempt == 2:
                    speak(
                        "Unable to set up Access Key. Set PICOVOICE_ACCESS_KEY in .env file and rerun setup."
                    )
                    print("ERROR: Failed to obtain valid Picovoice Access Key.")
                    logging.error(
                        "Failed to obtain Picovoice Access Key after max attempts"
                    )
                    # Ensure tts_engine is stopped before exiting
                    # This will be handled by the finally block in run_first_time_setup
                    # if "tts_engine" in globals() and tts_engine:
                    #     tts_engine.stop()
                    sys.exit(1)
        speak(
            "Download your .ppn model from the Picovoice Console for Windows and place it in 'wakeword_models/'."
        )
        print(
            "\nAvailable Picovoice Wake Words (ensure corresponding .ppn files are in 'wakeword_models/'):"
        )
        for i, ww_data in enumerate(AVAILABLE_WAKE_WORDS):
            print(f"{i + 1}. {ww_data['name']} (expects: {ww_data['model_file']})")
            if i < 5:
                speak(f"Option {i + 1}: {ww_data['name']}")
                time.sleep(0.3)

        print(f"\nINFO: Searching for .ppn models in {os.path.abspath(wakeword_models_dir)}...")
        found_ppn_files = glob.glob(os.path.join(wakeword_models_dir, "*.ppn"))

        if not found_ppn_files:
            speak("No .ppn model files found in the 'wakeword_models' directory. Please add your models and rerun setup.")
            print(f"ERROR: No .ppn files found in {os.path.abspath(wakeword_models_dir)}. Please place your Picovoice model file(s) there and run setup again.")
            logging.error(f"No .ppn files found in {wakeword_models_dir}")
            sys.exit(1)

        selectable_models = []
        for file_path in found_ppn_files:
            model_filename_base = os.path.splitext(os.path.basename(file_path))[0]
            display_name = model_filename_base # Default to filename
            # Try to use a more friendly name if it matches a predefined model
            for predefined in AVAILABLE_WAKE_WORDS:
                if os.path.normcase(os.path.normpath(predefined["model_file"])) == os.path.normcase(os.path.normpath(file_path)):
                    display_name = predefined["name"]
                    break
            selectable_models.append({"name": display_name, "model_file": file_path})

        selectable_models.sort(key=lambda x: x["name"]) # Sort for consistent display

        print("\nFound Picovoice Wake Word Models (select one):")
        for i, model_data in enumerate(selectable_models):
            print(f"{i + 1}. {model_data['name']} (from file: {os.path.basename(model_data['model_file'])})")
            if i < 5: # Speak only the first few options
                speak(f"Option {i + 1}: {model_data['name']}")
                time.sleep(0.3)

        chosen_picovoice_index = -1
        speak("Say the name or number of the Picovoice wake word.")
        time.sleep(0.5)
        chosen_picovoice_index = _get_choice_via_voice_input(
            prompt_context="Picovoice Wake Word",
            options=selectable_models, # Use the dynamically generated list
            key="name",
        )
        if chosen_picovoice_index == -1:
            speak("Trouble understanding your wake word choice. Try typing.")
            logging.info("Falling back to typed input for Picovoice wake word")
            for attempt in range(3):
                speak("Type the number of your chosen wake word.")
                try:
                    user_input = input(
                        "Enter the number for your chosen Picovoice wake word: "
                    )
                    choice = int(user_input)
                    if 1 <= choice <= len(selectable_models):
                        chosen_picovoice_index = choice - 1
                        break
                    else:
                        speak(
                            f"Invalid number. Choose between 1 and {len(selectable_models)}."
                        )
                        logging.warning("Invalid Picovoice wake word number input")
                except ValueError:
                    speak("Not a number. Try again.")
                    logging.warning("Non-numeric Picovoice wake word input")
                if attempt == 2 and chosen_picovoice_index == -1:
                    speak("Couldn't understand your choice. Try running setup again.")
                    logging.error(
                        "Failed to select Picovoice wake word after max attempts"
                    )
                    # Ensure tts_engine is stopped before exiting (handled by finally)
                    sys.exit(1)

        selected_picovoice_model_data = selectable_models[chosen_picovoice_index]
        model_path = selected_picovoice_model_data["model_file"]
        model_filename = os.path.basename(model_path)
        speak(
            f"You've selected {selected_picovoice_model_data['name']}. Verifying the model file {model_filename}."
        )
        logging.info(
            f"Selected Picovoice wake word: {selected_picovoice_model_data['name']}, model path: {model_path}"
        )
        print(f"\nVerifying your chosen Picovoice model: {model_filename} (Path: {model_path})")

        if not os.path.exists(model_path):
            error_message_console = (
                f"The Picovoice model file '{model_filename}' was not found at '{os.path.abspath(model_path)}'.\n"
                f"Please ensure you have downloaded it from the Picovoice Console (https://console.picovoice.ai/)\n"
                f"and placed it in the '{os.path.abspath(os.path.dirname(model_path))}' directory.\n"
                f"Then, please run this setup script again."
            )
            error_message_speak = (
                f"The model file {model_filename} was not found. "
                f"Please download it from the Picovoice Console, place it in the {os.path.dirname(model_path)} directory, "
                f"and then run this setup script again."
            )
            print(f"ERROR: {error_message_console}")
            speak(error_message_speak)
            logging.error(f"Picovoice model file not found: {model_path}. Full path checked: {os.path.abspath(model_path)}")
            sys.exit(1) # The finally block in run_first_time_setup will handle tts_engine.stop() if needed
        else:
            print(f"Model file '{model_filename}' found. Proceeding with testing.")
            logging.info(f"Picovoice model file found: {model_path}")

        speak(
            f"Loading and testing the {selected_picovoice_model_data['name']} wake word model."
        )
        try:
            from core.engine import VoiceCore

            test_core = VoiceCore(
                engine_type="picovoice",
                picovoice_keyword_paths=[model_path],
                picovoice_access_key=picovoice_access_key,
            )
            print("Picovoice engine initialized successfully with your model.")
            speak(
                f"The {selected_picovoice_model_data['name']} model loaded successfully."
            )
            logging.info(f"Picovoice model {model_filename} loaded successfully")
            test_core.stop()
        except Exception as e:
            print(f"Failed to initialize Picovoice engine with your model: {e}")
            speak(
                f"Problem loading {selected_picovoice_model_data['name']} model. Ensure '{model_filename}' is in 'wakeword_models/' and Access Key is valid. Try again."
            )
            logging.error(f"Failed to load Picovoice model {model_filename}: {e}")
            sys.exit(1)
        speak("Saving your configuration...")
        logging.info("Saving Picovoice configuration")
        config_data = DEFAULT_CONFIG.copy()
        config_data["first_run_complete"] = True
        config_data["chosen_wake_word_engine"] = "picovoice"
        config_data["chosen_wake_word_model_path"] = selected_picovoice_model_data[
            "model_file"
        ]
        config_data["picovoice_access_key_is_set_env"] = True
        if save_config(config_data):
            speak("Configuration saved successfully.")
            print("Configuration saved.")
            logging.info("Picovoice configuration saved")
        else:
            speak("Error saving configuration. Try again.")
            print("ERROR: Failed to save configuration.")
            logging.error("Failed to save Picovoice configuration")
            # Ensure tts_engine is stopped before exiting (handled by finally)
            sys.exit(1)
        return True
    except Exception as e_picovoice_setup:
        print(f"Unexpected error during Picovoice setup: {e_picovoice_setup}")
        speak("Unexpected error during Picovoice setup. Try again.")
        logging.error(f"Unexpected Picovoice setup error: {e_picovoice_setup}")
        # Ensure tts_engine is stopped before exiting (handled by finally)
        sys.exit(1)


def _setup_openwakeword_engine() -> bool:
    try:
        logging.info("Starting OpenWakeWord setup")
        speak(
            "You selected OpenWakeWord. ONNX models will be downloaded automatically. These support whispered wake words."
        )
        print("\n[OpenWakeWord Setup]")
        print(
            "- ONNX models will be downloaded to 'wakeword_models/'. If download fails, download manually."
        )
        download_openwakeword_models()
        speak("Choose a wake word from the following options.")
        print("\nAvailable OpenWakeWord Models:")
        for i, ww_data in enumerate(OPENWAKEWORD_MODELS):
            print(f"{i + 1}. {ww_data['name']}")
            speak(f"Option {i + 1}: {ww_data['name']}")
            time.sleep(0.3)
        chosen_index = -1
        speak("Say the name or number of the wake word.")
        time.sleep(0.5)
        chosen_index = _get_choice_via_voice_input(
            prompt_context="Wake Word", options=OPENWAKEWORD_MODELS, key="name"
        )
        if chosen_index == -1:
            speak("Trouble understanding your voice choice. Try typing.")
            logging.info("Falling back to typed input for OpenWakeWord wake word")
        retries = 3
        if chosen_index == -1:
            for attempt in range(retries):
                speak("Type the number of your choice.")
                try:
                    user_input = input("Enter the number for your chosen wake word: ")
                    choice = int(user_input)
                    if 1 <= choice <= len(OPENWAKEWORD_MODELS):
                        chosen_index = choice - 1
                        break
                    else:
                        speak(
                            f"Invalid number. Choose between 1 and {len(OPENWAKEWORD_MODELS)}."
                        )
                        logging.warning("Invalid OpenWakeWord wake word number input")
                except ValueError:
                    speak("Not a number. Try again.")
                    logging.warning("Non-numeric OpenWakeWord wake word input")
                if attempt < retries - 1:
                    speak("Let's try again.")
                else:
                    speak("Couldn't understand your choice. Try running setup again.")
                    logging.error(
                        "Failed to select OpenWakeWord wake word after max attempts"
                    )
                    # Ensure tts_engine is stopped before exiting (handled by finally)
                    sys.exit(1)
        selected_wake_word = OPENWAKEWORD_MODELS[chosen_index]
        speak(
            f"You've selected: {selected_wake_word['name']}. This model has a false accept rate below 0.5 per hour."
        )
        print(
            f"Selected wake word: {selected_wake_word['name']} (Model: {selected_wake_word['model_file']})"
        )
        logging.info(f"Selected OpenWakeWord wake word: {selected_wake_word['name']}")
        time.sleep(0.5)
        model_path_to_check = selected_wake_word["model_file"]
        if not os.path.exists(model_path_to_check):
            speak(
                f"Model file for {selected_wake_word['name']} not found at {model_path_to_check}. Download from {selected_wake_word.get('url', 'OpenWakeWord repository')}."
            )
            print(f"WARNING: Model file '{model_path_to_check}' not found!")
            logging.error(f"Model file not found: {model_path_to_check}")
            # Ensure tts_engine is stopped before exiting (handled by finally)
            sys.exit(1)
        else:
            print(f"Model file found at: {model_path_to_check}")
            logging.info(f"Model file found: {model_path_to_check}")
        try:
            from openwakeword.model import Model

            test_model = Model(wakeword_model_paths=[model_path_to_check])
            print(
                f"OpenWakeWord model '{selected_wake_word['name']}' loaded successfully."
            )
            speak(f"The {selected_wake_word['name']} model loaded successfully.")
            logging.info(
                f"OpenWakeWord model {model_path_to_check} loaded successfully"
            )
        except Exception as e:
            print(f"Failed to load OpenWakeWord model: {e}")
            speak(
                f"Error loading {selected_wake_word['name']} model. Ensure file at {model_path_to_check} is valid."
            )
            logging.error(
                f"Failed to load OpenWakeWord model {model_path_to_check}: {e}"
            )
            sys.exit(1)
        speak("Saving your configuration...")
        logging.info("Saving OpenWakeWord configuration")
        config_data = DEFAULT_CONFIG.copy()
        config_data["first_run_complete"] = True
        config_data["chosen_wake_word_engine"] = "openwakeword"
        config_data["chosen_wake_word_model_path"] = selected_wake_word["model_file"]
        config_data["picovoice_access_key_is_set_env"] = False
        if save_config(config_data):
            speak("Configuration saved successfully.")
            print("Configuration saved.")
            logging.info("OpenWakeWord configuration saved")
        else:
            speak("Error saving configuration. Try again.")
            print("ERROR: Failed to save configuration.")
            logging.error("Failed to save OpenWakeWord configuration")
            # Ensure tts_engine is stopped before exiting (handled by finally)
            sys.exit(1)
        return True
    except Exception as e_oww_setup:
        print(f"Unexpected error during OpenWakeWord setup: {e_oww_setup}")
        speak("Unexpected error during OpenWakeWord setup. Try again.")
        logging.error(f"Unexpected OpenWakeWord setup error: {e_oww_setup}")
        # Ensure tts_engine is stopped before exiting (handled by finally)
        sys.exit(1)


def run_first_time_setup():
    logging.info("Starting first run setup")
    speak(
        "Welcome! This is your first time running the voice assistant, or setup wasn't completed."
    )
    time.sleep(0.5)
    speak("Let's configure a few things.")
    time.sleep(0.5)
    try:
        engine_setup_successful = False
        engine_choice = select_wake_word_engine()
        if engine_choice == "picovoice":
            logging.info("Proceeding with Picovoice setup, skipping OpenWakeWord")
            engine_setup_successful = _setup_picovoice_engine()
        elif engine_choice == "openwakeword":
            logging.info("Proceeding with OpenWakeWord setup")
            engine_setup_successful = _setup_openwakeword_engine()
        if engine_setup_successful:
            tts_engine_type = select_tts_engine()
            _select_and_configure_tts_voice(tts_engine_type)
            time.sleep(0.5)
            speak("Setup complete! Restart the application to apply changes.")
            print("\nSetup complete. Please restart the main application.")
            logging.info("Setup completed successfully")
        else:
            speak("Wake word engine setup incomplete. Try running setup again.")
            logging.error("Wake word engine setup incomplete")
    finally:
        if "tts_engine" in globals() and tts_engine: # Check if tts_engine was successfully imported and initialized
            print("INFO: Stopping TTS engine before exiting.")
            logging.info("Stopping TTS engine")
            tts_engine.stop()


if __name__ == "__main__":
    run_first_time_setup()
