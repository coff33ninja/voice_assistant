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
    temp_stt_voice_core = None
    try:
        print("INFO: Initializing temporary VoiceCore for STT during TTS voice selection...")
        from core.engine import VoiceCore # Import here to ensure it's available
        temp_stt_voice_core = VoiceCore(engine_type="openwakeword", openwakeword_model_path=None) # STT-only mode

        for voice_attempt_tts in range(3):
            speak(f"TTS Voice Choice Attempt {voice_attempt_tts + 1}. Please speak now.")
            print("Listening for TTS voice choice (3 seconds)...")
            time.sleep(0.5)
            recorded_audio_frames_tts = []
            transcribed_text_tts = ""
            if temp_stt_voice_core and hasattr(temp_stt_voice_core, 'stream') and temp_stt_voice_core.stream and not temp_stt_voice_core.stream.is_stopped():
                for _ in range(int(16000 / 1280 * 3)): # Approx 3 seconds of audio
                    try:
                        audio_chunk_tts = temp_stt_voice_core.stream.read(1280, exception_on_overflow=False)
                        recorded_audio_frames_tts.append(audio_chunk_tts)
                    except IOError:
                        break
                print("TTS voice choice recording finished.")
                if recorded_audio_frames_tts:
                    full_audio_data_tts = b''.join(recorded_audio_frames_tts)
                    if hasattr(temp_stt_voice_core, 'whisper_model') and temp_stt_voice_core.whisper_model:
                        print("Transcribing TTS voice choice...")
                        transcribed_text_tts = transcribe_audio_with_whisper(temp_stt_voice_core.whisper_model, full_audio_data_tts)
                        if transcribed_text_tts:
                            print(f"Whisper transcribed TTS choice as: '{transcribed_text_tts}'")
                    else:
                        print("Whisper model not available in temporary VoiceCore instance.")
            if transcribed_text_tts:
                chosen_voice_obj_idx = match_choice_from_text(transcribed_text_tts, available_tts_voices, key="name")
                if chosen_voice_obj_idx != -1:
                    break
            speak("I didn't quite catch your voice selection.")
    finally:
        if temp_stt_voice_core:
            print("INFO: Stopping temporary VoiceCore for STT (TTS choice)...")
            temp_stt_voice_core.stop()

    if chosen_voice_obj_idx == -1: # Fallback to typed input
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

def run_first_time_setup():
    """
    Guides the user through the initial configuration of the voice assistant.
    
    This interactive setup process assists the user in selecting and verifying a wake word engine (Picovoice Porcupine or OpenWakeWord), configuring the appropriate wake word model, and choosing a text-to-speech (TTS) voice. The function handles both voice and typed input for selections, verifies model files, saves configuration settings, and provides spoken and printed feedback throughout. On completion or critical failure, the function exits the script after cleanup.
    """
    speak("Welcome! It looks like this is your first time running the voice assistant, or your setup wasn't completed.")
    time.sleep(0.5)
    speak("Let's get a few things configured.")
    time.sleep(0.5)

    # 1. Wake word engine selection
    engine_choice = select_wake_word_engine()
    if engine_choice == 'picovoice':
        speak("You selected Picovoice Porcupine. Please download your .ppn model from the Picovoice Console for your platform (Windows). Place it in the 'wakeword_models/' directory in your project root.")
        print("\n[Picovoice Setup]")
        print("- Download your .ppn model from https://console.picovoice.ai/")
        print("- Place the downloaded .ppn file in the 'wakeword_models/' directory in your project root.")
        while True:
            model_filename = input("Enter the exact filename of your .ppn model (e.g., Jarvis.ppn): ").strip()
            model_path = os.path.join("wakeword_models", model_filename)
            if os.path.exists(model_path):
                print(f"Model file found at: {model_path}")
                speak(f"Model file {model_filename} found. Ready to test.")
                break
            else:
                print(f"Model file '{model_path}' not found. Please ensure you have placed it in the 'wakeword_models/' directory.")
                speak(f"Model file {model_filename} not found. Please try again.")
        # Ask user to test the wake word
        print("\nLet's test your wake word model before continuing.")
        speak("Let's test your wake word model. Please say your wake word now.")
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
            speak("Wake word model loaded successfully. If you hear this, your model works.")
            test_core.stop()
        except Exception as e:
            print(f"Failed to initialize Picovoice engine with your model: {e}")
            speak("There was a problem loading your wake word model. Please check the file and try again.")
            sys.exit(1)
        # Set up the config for Picovoice
        selected_wake_word = {"name": os.path.splitext(model_filename)[0], "model_file": model_path}
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
                tts_engine.stop()
            sys.exit(1)
        # Proceed directly to TTS voice selection
        _select_and_configure_tts_voice()

        time.sleep(0.5)
        speak("Setup is complete! Please restart the main application for the changes to take effect.")
        print("\nSetup complete. Please restart the main application.")
        if 'tts_engine' in globals():
            tts_engine.stop()
        sys.exit(0)
    else:
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
        speak("You can say the name of the wake word, or its number.")
        time.sleep(0.5)
        temp_voice_core = None # Define before try block
        try:
            print("INFO: Initializing temporary VoiceCore for STT during setup...")
            from core.engine import VoiceCore
            temp_voice_core = VoiceCore(engine_type="openwakeword", openwakeword_model_path=None)

            for voice_attempt in range(3): # Max 3 voice attempts
                speak(f"Attempt {voice_attempt + 1}. Please say your choice now.")
                print("Listening for your choice... (Speak now)")
                # Manually simulate command listening for STT part of VoiceCore
                # This is a HACK and relies on internal VoiceCore behavior/structure.
                # A proper VoiceCore.transcribe_once() method would be much better.

                # --- Actual Voice Input Capture & Transcribe ---
                transcribed_text = "" # Ensure it's defined
                speak("Please say your choice clearly after the beep... (beep sound not implemented yet)")
                print("Listening for 3 seconds...")
                time.sleep(0.5) # Brief pause for user to prepare

                recorded_audio_frames = []
                try:
                    if temp_voice_core and hasattr(temp_voice_core, 'stream') and temp_voice_core.stream and not temp_voice_core.stream.is_stopped():
                        for _ in range(int(16000 / 1280 * 3)):
                            try:
                                audio_chunk = temp_voice_core.stream.read(1280, exception_on_overflow=False)
                                recorded_audio_frames.append(audio_chunk)
                            except IOError as e_read:
                                print(f"Error reading audio stream during setup: {e_read}")
                                break
                        print("Finished recording.")

                        if recorded_audio_frames:
                            full_audio_data = b''.join(recorded_audio_frames)
                            if hasattr(temp_voice_core, 'whisper_model') and temp_voice_core.whisper_model:
                                print("Transcribing your choice...")
                                transcribed_text = transcribe_audio_with_whisper(temp_voice_core.whisper_model, full_audio_data)
                                if transcribed_text:
                                    print(f"Whisper transcribed: '{transcribed_text}'")
                            else:
                                print("Whisper model not available in temporary VoiceCore instance.")
                        else:
                            speak("I didn't capture any audio.")
                    else:
                        print("Temporary VoiceCore stream not available for recording.")
                        speak("Sorry, I couldn't access the microphone for voice input.")
                except Exception as e_voice_record:
                    print(f"An error occurred during voice capture: {e_voice_record}")
                    speak("An unexpected error occurred with voice input.")
                # --- End Actual Voice Input ---

                if transcribed_text:
                    print(f"I heard: {transcribed_text}")
                    chosen_index = match_choice_from_text(transcribed_text, OPENWAKEWORD_MODELS, key="name") # Match against OPENWAKEWORD_MODELS
                    if chosen_index != -1:
                        break  # Found a match by name or number pattern

                if chosen_index != -1:
                    break # Break from voice_attempt loop
                speak("I didn't quite catch that.")
            if chosen_index != -1:
                speak("Okay, I got your choice by voice!")
            else:
                speak("I'm having a bit of trouble understanding your voice choice. Let's try with typed input.")
        except Exception as e_voice_init:
            print(f"Error during temporary VoiceCore init/STT for setup: {e_voice_init}")
            speak("There was an issue setting up voice input. We'll use typed input instead.")
        finally:
            if temp_voice_core:
                # Attempt to clean up VoiceCore resources if it was initialized
                # VoiceCore.stop() should handle PyAudio, threads, etc.
                # This is important if VoiceCore started its PyAudio stream.
                # The current hacky VoiceCore init with openwakeword_model_path=None might not fully start it,
                # but calling stop() should be safe if it's implemented robustly.
                try:
                    print("INFO: Stopping temporary VoiceCore for STT...")
                    temp_voice_core.stop()
                except Exception as e_stop_temp_vc:
                    print(f"Error stopping temporary VoiceCore: {e_stop_temp_vc}")
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
                        tts_engine.stop()
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
                tts_engine.stop()
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
                tts_engine.stop()
            sys.exit(1)

        # --- TTS Voice Selection ---
        _select_and_configure_tts_voice()

        time.sleep(0.5)
        speak("Setup is complete! Please restart the main application for the changes to take effect.")
        print("\nSetup complete. Please restart the main application.")

        # Gracefully stop TTS engine
        if 'tts_engine' in globals():
            tts_engine.stop()

if __name__ == "__main__":
    run_first_time_setup()
