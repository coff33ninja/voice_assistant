"""
first_run_setup.py

Guides the user through initial configuration of the voice assistant, including:
- Picovoice Access Key setup
- Wake word selection (voice or typed)
- Wake word model file verification
- TTS voice selection (voice or typed)
- Configuration saving

This script is intended to be run as a standalone setup utility.
"""

import os
import sys
import time # For small delays
import re  # For text matching
import numpy as np
import urllib.request
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Ensure core modules can be imported if script is run directly from root
# This might be needed if first_run_setup.py is run as a separate process
if __name__ == '__main__':
    # Get the absolute path of the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Add the parent directory (project root) to sys.path if it's not already there
    project_root = script_dir # Assuming script is in project root
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

try:
    from core.tts import speak, tts_engine # Import tts_engine to stop it later
    from core.user_config import load_config, save_config, DEFAULT_CONFIG
    # from core.engine import VoiceCore # For STT (moved to where needed)
except ImportError as e:
    print(f"Error: Could not import core modules: {e}")
    print("Please ensure that the script is run from the project's root directory,")
    print("or that the project root is in your PYTHONPATH.")
    sys.exit(1)

# --- Placeholder Wake Word Data ---
# DEVELOPER: Replace these with your actual Picovoice .ppn files and desired names.
# Ensure these .ppn files are placed in the 'wakeword_models/' directory in the project root.
AVAILABLE_WAKE_WORDS = [
    {"name": "Computer", "model_file": "wakeword_models/Computer.ppn"}, # Example, replace .ppn
    {"name": "Jarvis", "model_file": "wakeword_models/Jarvis.ppn"},   # Example, replace .ppn
    {"name": "Assistant", "model_file": "wakeword_models/Assistant.ppn"} # Example, replace .ppn
]
# --- End Placeholder Wake Word Data ---

# --- OpenWakeWord Model Data ---
# List of OpenWakeWord models to download if not present
OPENWAKEWORD_MODELS = [
    {"name": "Computer", "url": "https://github.com/synesthesiam/openwakeword-models/raw/main/onnx/computer.onnx", "model_file": "wakeword_models/Computer.onnx"},
    {"name": "Jarvis", "url": "https://github.com/synesthesiam/openwakeword-models/raw/main/onnx/jarvis.onnx", "model_file": "wakeword_models/Jarvis.onnx"},
    {"name": "Assistant", "url": "https://github.com/synesthesiam/openwakeword-models/raw/main/onnx/assistant.onnx", "model_file": "wakeword_models/Assistant.onnx"},
]

def download_openwakeword_models():
    """
    Ensures all required OpenWakeWord ONNX models are present in the wakeword_models directory.
    
    Downloads each model from its specified URL if it does not already exist locally. If a download fails, instructs the user to manually download the missing model.
    """
    if not os.path.exists("wakeword_models"):
        os.makedirs("wakeword_models")
    for model in OPENWAKEWORD_MODELS:
        path = model["model_file"]
        url = model["url"]
        if not os.path.exists(path):
            print(f"Downloading OpenWakeWord model for {model['name']}...")
            try:
                urllib.request.urlretrieve(url, path)
                print(f"Downloaded: {path}")
            except Exception as e:
                print(f"Unable to download {url}: {e}")
                print(f"Please download the ONNX model for '{model['name']}' manually from {url} and place it at {path}.")
        else:
            print(f"Model already exists: {path}")

def transcribe_audio_with_whisper(whisper_model, audio_bytes) -> str:
    """
    Transcribes raw audio bytes to text using a Whisper model.
    
    Args:
        whisper_model: An instance of a Whisper speech-to-text model.
        audio_bytes: Raw audio data in 16-bit PCM format.
    
    Returns:
        The transcribed text in lowercase, or an empty string if transcription fails.
    """
    try:
        audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
        audio_float32 = audio_int16.astype(np.float32) / 32768.0
        result = whisper_model.transcribe(audio_float32, fp16=False)
        transcribed_text = result['text']
        if isinstance(transcribed_text, list):
            transcribed_text = " ".join(str(x) for x in transcribed_text)
        return transcribed_text.strip().lower() if transcribed_text else ""
    except Exception as e:
        print(f"Error during Whisper transcription: {e}")
        return ""

def match_choice_from_text(transcribed_text: str, options: list[dict], key: str = "name") -> int:
    """
    Matches transcribed user input to an option index based on name or numeric reference.
    
    Args:
        transcribed_text: The user's spoken or typed input to interpret.
        options: List of option dictionaries to match against.
        key: The dictionary key to use for name matching (default is "name").
    
    Returns:
        The index of the matched option, or -1 if no suitable match is found.
    """
    if not transcribed_text:
        return -1
    text = transcribed_text.strip().lower()
    for i, opt in enumerate(options):
        # Match by name (case-insensitive, substring)
        if opt[key].lower() in text:
            return i
        # Match by number (e.g., "number one", "option 1", "1")
        if re.search(r'(number|option|#)?\s*' + str(i + 1) + r'\b', text, re.IGNORECASE) or text == str(i + 1):
            return i
    # Also try matching number words (e.g., "one", "two")
    number_words = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
    for i, opt in enumerate(options):
        if i < len(number_words) and number_words[i] in text:
            return i
    return -1

# --- Wake Word Engine Selection ---
def select_wake_word_engine():
    """
    Prompts the user to select a wake word engine and returns the chosen engine type.
    
    Returns:
        The string 'picovoice' if Picovoice Porcupine is selected, or 'openwakeword' if OpenWakeWord is selected.
    """
    print("\nChoose your wake word engine:")
    print("1. Picovoice Porcupine (.ppn, requires Access Key, high accuracy, closed-source)")
    print("2. OpenWakeWord (.onnx, open-source, no key required)")
    while True:
        choice = input("Enter 1 for Picovoice or 2 for OpenWakeWord: ").strip()
        if choice == '1':
            return 'picovoice'
        elif choice == '2':
            return 'openwakeword'
        else:
            print("Invalid input. Please enter 1 or 2.")

def _get_choice_via_voice_input(prompt_context: str, options: list[dict], key: str = "name", attempts: int = 3, listen_duration_sec: int = 3) -> int:
    """
    Handles getting a user's choice from a list of options via voice input.

    Args:
        prompt_context: A string describing what the user is choosing (e.g., "Wake Word", "TTS Voice").
        options: The list of dictionary options to choose from.
        key: The dictionary key in `options` to use for matching (default "name").
        attempts: Maximum number of voice input attempts.
        listen_duration_sec: Duration in seconds to listen for each attempt.

    Returns:
        The index of the chosen option in the `options` list, or -1 if no choice was successfully made.
    """
    chosen_index = -1
    temp_voice_core = None
    try:
        print(f"INFO: Initializing temporary VoiceCore for STT ({prompt_context} selection)...")
        from core.engine import VoiceCore # Import here to ensure it's available
        temp_voice_core = VoiceCore(engine_type="openwakeword", openwakeword_model_path=None) # STT-only mode

        for attempt_num in range(attempts):
            speak(f"{prompt_context} Choice Attempt {attempt_num + 1}. Please speak now.")
            print(f"Listening for {prompt_context.lower()} choice ({listen_duration_sec} seconds)...")
            time.sleep(0.5) # Brief pause for user readiness

            recorded_audio_frames = []
            transcribed_text = ""

            if temp_voice_core and hasattr(temp_voice_core, 'stream') and temp_voice_core.stream and not temp_voice_core.stream.is_stopped():
                # VoiceCore stream is hardcoded to 1280 frames_per_buffer, 16000 Hz
                for _ in range(int(16000 / 1280 * listen_duration_sec)):
                    try:
                        audio_chunk = temp_voice_core.stream.read(1280, exception_on_overflow=False)
                        recorded_audio_frames.append(audio_chunk)
                    except IOError as e_read:
                        print(f"Error reading audio stream during {prompt_context} selection: {e_read}")
                        break # Stop trying to read for this attempt
                print(f"{prompt_context} choice recording finished.")

                if recorded_audio_frames:
                    full_audio_data = b''.join(recorded_audio_frames)
                    if hasattr(temp_voice_core, 'whisper_model') and temp_voice_core.whisper_model:
                        print(f"Transcribing {prompt_context.lower()} choice...")
                        transcribed_text = transcribe_audio_with_whisper(temp_voice_core.whisper_model, full_audio_data)
                        if transcribed_text:
                            print(f"Whisper transcribed {prompt_context.lower()} choice as: '{transcribed_text}'")
            if transcribed_text:
                chosen_index = match_choice_from_text(transcribed_text, options, key=key)
                if chosen_index != -1:
                    speak(f"Okay, I understood your {prompt_context.lower()} choice by voice!")
                    break # Successfully matched
            
            if chosen_index == -1 and attempt_num < attempts - 1: # If not matched and not the last attempt
                speak(f"I didn't quite catch your {prompt_context.lower()} selection.")
    finally:
        if temp_voice_core:
            print(f"INFO: Stopping temporary VoiceCore for STT ({prompt_context} selection)...")
            temp_voice_core.stop()
    return chosen_index

def _select_and_configure_tts_voice():
    """
    Handles the TTS voice selection process, including voice/typed input and configuration saving.
    """
    speak("Next, let's choose a voice for me to speak with.")
    time.sleep(0.5)
    available_tts_voices = []
    if 'tts_engine' in globals() and hasattr(tts_engine, 'get_available_voices'):
        available_tts_voices = tts_engine.get_available_voices()

    if not available_tts_voices:
        speak("I couldn't find any specific voices to choose from right now, so we'll stick with the default.")
        print("INFO: No TTS voices found or tts_engine not available for get_available_voices.")
        return # Nothing more to do if no voices

    speak("Here are the voices I can use:")
    print("\nAvailable TTS Voices:")
    for i, voice in enumerate(available_tts_voices):
        voice_display_name = voice.get('name', f"Voice {i+1}")
        print(f"{i + 1}. {voice_display_name}")
        if i < 5: # Limit spoken options
            speak(f"Option {i + 1}: {voice_display_name}")
            time.sleep(0.2)
    if len(available_tts_voices) > 5:
        speak("There are more voices listed in the console if you'd like to see them all.")

    chosen_voice_id = None
    chosen_voice_obj_idx = -1

    speak("You can say the name or number of the voice you'd like.")
    time.sleep(0.5)
    
    chosen_voice_obj_idx = _get_choice_via_voice_input(
        prompt_context="TTS Voice",
        options=available_tts_voices,
        key="name"
    )

    if chosen_voice_obj_idx == -1: # Fallback to typed input
        speak("I'm having a bit of trouble understanding your voice choice for TTS. Let's try with typed input.")
        speak("Let's try selecting the voice with typed input.")
        for tts_attempt_typed in range(3):
            speak("Please type the number of your chosen voice.")
            try:
                user_input_tts = input("Enter the number for your chosen TTS voice: ")
                choice_tts = int(user_input_tts)
                if 1 <= choice_tts <= len(available_tts_voices):
                    chosen_voice_obj_idx = choice_tts - 1
                    break
                else:
                    speak("That's not a valid number for voice choice.")
            except ValueError:
                speak("That didn't seem like a number for voice choice.")
            if tts_attempt_typed == 2 and chosen_voice_obj_idx == -1: # Check if still not chosen after last attempt
                speak("Skipping TTS voice selection for now.")

    if chosen_voice_obj_idx != -1:
        chosen_voice_id = available_tts_voices[chosen_voice_obj_idx]['id']
        chosen_voice_name = available_tts_voices[chosen_voice_obj_idx]['name']
        speak(f"You've selected {chosen_voice_name} as your voice. Nice!")
        print(f"Selected TTS Voice: {chosen_voice_name} (ID: {chosen_voice_id})")
        current_config = load_config() # Load potentially already saved config
        current_config["chosen_tts_voice_id"] = chosen_voice_id
        if save_config(current_config):
            print("TTS Voice ID saved to configuration.")
            if 'tts_engine' in globals() and hasattr(tts_engine, 'set_voice'):
                if tts_engine.set_voice(chosen_voice_id):
                    speak(f"I will now try to use the {chosen_voice_name} voice.")
                else: # If set_voice failed or tts_engine not fully ready
                    speak(f"I'll use the {chosen_voice_name} voice the next time you start me.")
        else:
            speak("There was an error saving your voice choice.")
    else: # chosen_voice_obj_idx remained -1
        speak("Okay, we'll stick with the default voice for now.")
        print("INFO: Default TTS voice will be used.")

def _setup_picovoice_engine() -> bool:
    """
    Handles the setup process specific to the Picovoice engine.
    Returns True if setup is successful and configuration is saved, False otherwise (or exits).
    """
    try:
        speak("You selected Picovoice Porcupine. Please download your .ppn model from the Picovoice Console for your platform (Windows). Place it in the 'wakeword_models/' directory in your project root.")
        speak("Ensure the model files listed correspond to what you have placed in the 'wakeword_models' directory.")
        print("\n[Picovoice Setup]")
        print("- Download your .ppn model from https://console.picovoice.ai/")
        print("- Place the downloaded .ppn file in the 'wakeword_models/' directory in your project root.")
        print("\nAvailable Picovoice Wake Words (ensure corresponding .ppn files are in 'wakeword_models/'):")
        for i, ww_data in enumerate(AVAILABLE_WAKE_WORDS):
            print(f"{i + 1}. {ww_data['name']} (expects: {ww_data['model_file']})")
            if i < 5: # Speak first few
                speak(f"Option {i + 1}: {ww_data['name']}")
                time.sleep(0.3)

        chosen_picovoice_index = -1
        speak("You can say the name or number of the Picovoice wake word you have prepared.")
        time.sleep(0.5)

        chosen_picovoice_index = _get_choice_via_voice_input(
            prompt_context="Picovoice Wake Word",
            options=AVAILABLE_WAKE_WORDS,
            key="name"
        )

        if chosen_picovoice_index == -1: # Fallback to typed input
            speak("I'm having trouble understanding your Picovoice wake word choice. Let's try with typed input.")
            for attempt in range(3):
                speak("Please type the number of your chosen Picovoice wake word.")
                try:
                    user_input = input("Enter the number for your chosen Picovoice wake word: ")
                    choice = int(user_input)
                    if 1 <= choice <= len(AVAILABLE_WAKE_WORDS):
                        chosen_picovoice_index = choice - 1
                        break
                    else:
                        speak(f"That's not a valid number. Please choose between 1 and {len(AVAILABLE_WAKE_WORDS)}.")
                except ValueError:
                    speak("That didn't seem like a number. Please try again.")
                if attempt == 2 and chosen_picovoice_index == -1:
                    speak("Sorry, I couldn't understand your choice. Please try running the setup again.")
                    if 'tts_engine' in globals():
                        tts_engine.stop()
                    sys.exit(1)
        
        selected_picovoice_model_data = AVAILABLE_WAKE_WORDS[chosen_picovoice_index]
        model_path = selected_picovoice_model_data['model_file']
        model_filename = os.path.basename(model_path) # Get filename for messages
        speak(f"You've selected {selected_picovoice_model_data['name']}. I will check for the file {model_filename}.")

        # Ask user to test the wake word
        print(f"\nVerifying and testing your chosen Picovoice model: {model_filename}")
        speak(f"Now, I'll try to load and test the {selected_picovoice_model_data['name']} wake word model.")
        # Minimal test: try to initialize Picovoice engine with the model
        try:
            from core.engine import VoiceCore  # Import here to avoid unused import warning
            picovoice_access_key = os.environ.get("PICOVOICE_ACCESS_KEY")
            if not picovoice_access_key:
                raise RuntimeError("Picovoice access key not found in environment. Please set PICOVOICE_ACCESS_KEY in your .env file or environment variables.")
            test_core = VoiceCore(
                engine_type="picovoice",
                picovoice_keyword_paths=[model_path],
                picovoice_access_key=picovoice_access_key
            )
            print("Picovoice engine initialized successfully with your model.")
            speak(f"The {selected_picovoice_model_data['name']} model loaded successfully.")
            test_core.stop()
        except Exception as e:
            print(f"Failed to initialize Picovoice engine with your model: {e}")
            speak(f"There was a problem loading the {selected_picovoice_model_data['name']} model. Please ensure the file '{model_filename}' is correctly placed in 'wakeword_models/' and your Picovoice Access Key is valid. Then, try the setup again.")
            sys.exit(1)
        # Set up the config for Picovoice
        selected_wake_word = selected_picovoice_model_data # Contains 'name' and 'model_file'
        # Save configuration and skip generic wake word selection
        speak("Saving your configuration...")
        config_data = DEFAULT_CONFIG.copy() # Start with defaults
        config_data["first_run_complete"] = True
        config_data["chosen_wake_word_engine"] = "picovoice"
        config_data["chosen_wake_word_model_path"] = selected_wake_word["model_file"]
        config_data["picovoice_access_key_is_set_env"] = True
        if save_config(config_data):
            speak("Configuration saved successfully.")
            print("Configuration saved.")
        else:
            speak("There was an error saving your configuration. Please try again.")
            print("ERROR: Failed to save configuration.")
            if 'tts_engine' in globals():
                tts_engine.stop() # Ensure TTS stops before exit
            sys.exit(1)
        return True # Picovoice setup and config save successful
    except Exception as e_picovoice_setup: # Catch any unexpected error during picovoice setup
        print(f"An unexpected error occurred during Picovoice setup: {e_picovoice_setup}")
        speak("An unexpected error occurred during the Picovoice setup. Please try again.")
        if 'tts_engine' in globals():
            tts_engine.stop()
        sys.exit(1)

def _setup_openwakeword_engine() -> bool:
    """
    Handles the setup process specific to the OpenWakeWord engine.
    Returns True if setup is successful and configuration is saved, False otherwise (or exits).
    """
    try:
        speak("You selected OpenWakeWord. ONNX models will be downloaded automatically if possible.")
        print("\n[OpenWakeWord Setup]")
        print("- ONNX models will be downloaded to 'wakeword_models/'. If download fails, download manually from the URLs printed above.")
        download_openwakeword_models()

        # --- Wake Word Selection for OpenWakeWord only ---
        # 2. Choose a Wake Word
        speak("Please choose a wake word from the following options.")
        print("\nAvailable OpenWakeWord Models:")
        for i, ww_data in enumerate(OPENWAKEWORD_MODELS): # Use OPENWAKEWORD_MODELS for choices
            print(f"{i + 1}. {ww_data['name']}")
            speak(f"Option {i + 1}: {ww_data['name']}") # Speak names from OPENWAKEWORD_MODELS
            time.sleep(0.3)

        chosen_index = -1 # Ensure it's initialized before voice or typed input attempts

        # --- Voice Input for Wake Word Selection ---
        speak("You can say the name of the wake word, or its number from the list.")
        time.sleep(0.5)
        
        chosen_index = _get_choice_via_voice_input(
            prompt_context="Wake Word",
            options=OPENWAKEWORD_MODELS,
            key="name"
        )
        if chosen_index == -1: # If voice input failed
            speak("I'm having a bit of trouble understanding your voice choice. Let's try with typed input.")
        # --- End Voice Input Logic ---

        retries = 3
        # Fallback to typed input if voice selection failed
        if chosen_index == -1:
            for attempt in range(retries):
                speak("Please type the number of your choice.")
                try:
                    user_input = input("Enter the number for your chosen wake word: ")
                    choice = int(user_input)
                    if 1 <= choice <= len(OPENWAKEWORD_MODELS): # Check against length of OPENWAKEWORD_MODELS
                        chosen_index = choice - 1
                        break
                    else:
                        speak(f"That's not a valid number. Please choose between 1 and {len(OPENWAKEWORD_MODELS)}.")
                except ValueError:
                    speak("That didn't seem like a number. Please try again.") # Corrected speak message

                if attempt < retries - 1:
                    speak("Let's try that again.")
                else:
                    speak("Sorry, I'm having trouble understanding your choice. Please try running the setup again later.")
                    if 'tts_engine' in globals():
                        tts_engine.stop() # Ensure TTS stops before exit
                    sys.exit(1)

        selected_wake_word = OPENWAKEWORD_MODELS[chosen_index] # Get selection from OPENWAKEWORD_MODELS
        speak(f"You've selected: {selected_wake_word['name']}. Great choice!")
        print(f"Selected wake word: {selected_wake_word['name']} (Model: {selected_wake_word['model_file']})")
        time.sleep(0.5)

        # 4. Verify model file existence (basic check)
        # Assumes script is run from project root where wakeword_models/ is located.
        model_path_to_check = selected_wake_word['model_file']
        if not os.path.exists(model_path_to_check):
            speak(f"Warning: The model file for {selected_wake_word['name']} at {model_path_to_check} was not found.")
            print(f"WARNING: Model file '{model_path_to_check}' not found! It should have been downloaded automatically. Please check the 'wakeword_models' directory.")
            speak("Please check the file path and run the setup again.")
            if 'tts_engine' in globals():
                tts_engine.stop() # Ensure TTS stops before exit
            sys.exit(1)
        else:
            print(f"Model file found at: {model_path_to_check}")


        # 5. Save Configuration
        speak("Saving your configuration...")
        config_data = DEFAULT_CONFIG.copy() # Start with defaults
        config_data["first_run_complete"] = True
        config_data["chosen_wake_word_engine"] = "openwakeword" # Correct engine type
        config_data["chosen_wake_word_model_path"] = selected_wake_word["model_file"]
        config_data["picovoice_access_key_is_set_env"] = False # Not relevant for OpenWakeWord
        
        if save_config(config_data):
            speak("Configuration saved successfully.")
            print("Configuration saved.")
        else:
            speak("There was an error saving your configuration. Please try again.")
            print("ERROR: Failed to save configuration.")
            if 'tts_engine' in globals():
                tts_engine.stop() # Ensure TTS stops before exit
            sys.exit(1)
        return True # OpenWakeWord setup and config save successful
    except Exception as e_oww_setup: # Catch any unexpected error during oww setup
        print(f"An unexpected error occurred during OpenWakeWord setup: {e_oww_setup}")
        speak("An unexpected error occurred during the OpenWakeWord setup. Please try again.")
        if 'tts_engine' in globals():
            tts_engine.stop()
        sys.exit(1)

def run_first_time_setup():
    """
    Guides the user through the initial configuration of the voice assistant.
    
    This interactive setup process assists the user in selecting and verifying a wake word engine (Picovoice Porcupine or OpenWakeWord), configuring the appropriate wake word model, and choosing a text-to-speech (TTS) voice. The function handles both voice and typed input for selections, verifies model files, saves configuration settings, and provides spoken and printed feedback throughout. On completion or critical failure, the function exits the script after cleanup.
    """
    speak("Welcome! It looks like this is your first time running the voice assistant, or your setup wasn't completed.")
    time.sleep(0.5)
    speak("Let's get a few things configured.")
    time.sleep(0.5)

    engine_setup_successful = False
    engine_choice = select_wake_word_engine()

    if engine_choice == 'picovoice':
        engine_setup_successful = _setup_picovoice_engine()
    elif engine_choice == 'openwakeword': # Explicitly check for 'openwakeword'
        engine_setup_successful = _setup_openwakeword_engine()
    
    if engine_setup_successful:
        _select_and_configure_tts_voice()
        time.sleep(0.5)
        speak("Setup is complete! Please restart the main application for the changes to take effect.")
        print("\nSetup complete. Please restart the main application.")
    else: # Should not be reached if sub-functions sys.exit() on critical failure, but as a fallback
        speak("Wake word engine setup was not completed. Please try running the setup again.")

        # Gracefully stop TTS engine
        if 'tts_engine' in globals():
            tts_engine.stop()

if __name__ == "__main__":
    run_first_time_setup()
