import os
import sys
import time # For small delays

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
    from core.engine import VoiceCore # For STT
    import re # For text matching
    import numpy as np
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

def run_first_time_setup():
    speak("Welcome! It looks like this is your first time running the voice assistant, or your setup wasn't completed.")
    time.sleep(0.5)
    speak("Let's get a few things configured.")
    time.sleep(0.5)

    # 1. Check for Picovoice Access Key
    picovoice_access_key = os.environ.get("PICOVOICE_ACCESS_KEY")
    if not picovoice_access_key:
        speak("To use custom wake words with Picovoice, I need an Access Key.")
        print("\nIMPORTANT: Picovoice Access Key Not Found!")
        print("Please set the 'PICOVOICE_ACCESS_KEY' environment variable with your key from the Picovoice Console.")
        print("You can get a free key at https://console.picovoice.ai/")
        print("After setting the environment variable, please restart the application.")
        # Stop TTS engine before exiting
        if 'tts_engine' in globals(): tts_engine.stop()
        sys.exit(1)
    else:
        print(f"Picovoice Access Key found in environment.")
        speak("Great, I found your Picovoice Access Key.")
        time.sleep(0.5)

    # 2. Choose a Wake Word
    speak("Please choose a wake word from the following options.")
    print("\nAvailable Wake Words:")
    for i, ww_data in enumerate(AVAILABLE_WAKE_WORDS):
        print(f"{i + 1}. {ww_data['name']}")
        speak(f"Option {i + 1}: {ww_data['name']}")
        time.sleep(0.3)

    chosen_index = -1 # Ensure it's initialized before voice or typed input attempts

    # --- Voice Input for Wake Word Selection ---
    speak("You can say the name of the wake word, or its number.")
    time.sleep(0.5)
    temp_voice_core = None # Define before try block
    try:
        # Initialize VoiceCore for STT only (no wake word, default whisper model)
        # Picovoice params not needed here as we are not using its wake word functionality
        print("INFO: Initializing temporary VoiceCore for STT during setup...")
        # Hack: Provide None for openwakeword_model_path to rely on Whisper only.
        # This assumes VoiceCore's __init__ can handle openwakeword_model_path=None
        # and still initialize Whisper. This is a potential point of failure if VoiceCore changes.
        # A dedicated STT-only mode or class would be cleaner.
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
                # temp_voice_core should be initialized before this loop from previous step's logic
                if temp_voice_core and hasattr(temp_voice_core, 'stream') and temp_voice_core.stream and not temp_voice_core.stream.is_stopped():
                    # PyAudio stream parameters from VoiceCore: CHANNELS=1, RATE=16000, FRAMES_PER_BUFFER=1280
                    # Record for approx 3 seconds: (16000 rate / 1280 frames_per_buffer) * 3 seconds = approx 37.5 reads
                    for _ in range(int(16000 / 1280 * 3)): # Approx 3 seconds of audio
                        try:
                            audio_chunk = temp_voice_core.stream.read(1280, exception_on_overflow=False)
                            recorded_audio_frames.append(audio_chunk)
                        except IOError as e_read:
                            print(f"Error reading audio stream during setup: {e_read}")
                            break # Stop recording on error
                    print("Finished recording.")

                    if recorded_audio_frames:
                        full_audio_data = b''.join(recorded_audio_frames)
                        if hasattr(temp_voice_core, 'whisper_model') and temp_voice_core.whisper_model:
                            print("Transcribing your choice...")
                            # Whisper expects a NumPy array or file path. Convert bytes to NumPy array.
                            # This requires numpy. For now, this part is also a simplification.
                            # VoiceCore._process_command() has the full conversion and transcription logic.
                            # Replicating it here is complex. A direct STT method in VoiceCore is better.
                            # For this subtask, we'll assume temp_voice_core can provide transcription directly.
                            # This is a known simplification and will need a VoiceCore.transcribe(audio_bytes) method.
                            try:
                                # ---- This is the Idealized Call (requires VoiceCore modification) ----
                                # transcribed_text = temp_voice_core.transcribe_audio_bytes(full_audio_data)
                                # ---- Fallback: Manual Whisper call (if numpy is available and model loaded) ----
                                # import numpy as np # Would need to be added to imports
                                # audio_np = np.frombuffer(full_audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                                # result = temp_voice_core.whisper_model.transcribe(audio_np, fp16=False) # fp16 might need check
                                # transcribed_text = result['text'].strip().lower() if result else ''
                                # ---- Actual STT using Whisper ---
                                print("Transcribing your choice using Whisper... This may take a moment.")
                                try:
                                    # Convert audio bytes to NumPy array suitable for Whisper
                                    # Whisper expects 16kHz mono audio, float32, normalized to [-1.0, 1.0]
                                    audio_int16 = np.frombuffer(full_audio_data, dtype=np.int16)
                                    audio_float32 = audio_int16.astype(np.float32) / 32768.0

                                    # Transcribe using the whisper_model from the temporary VoiceCore instance
                                    # Ensure whisper_model was loaded: check 'hasattr(temp_voice_core, 'whisper_model') and temp_voice_core.whisper_model'
                                    # This was checked before this block in the original structure.
                                    result = temp_voice_core.whisper_model.transcribe(audio_float32, fp16=False) # fp16=False for broader CPU compatibility
                                    transcribed_text = result['text'].strip().lower() if result and 'text' in result else ""
                                    if transcribed_text:
                                        print(f"Whisper transcribed: '{transcribed_text}'")
                                    else:
                                        print("Whisper could not transcribe the audio or returned empty text.")
                                        speak("I couldn't understand what you said.")
                                except AttributeError as ae:
                                    # This might happen if temp_voice_core or whisper_model is None
                                    print(f"STT Error: Whisper model not available or not loaded correctly in temp_voice_core: {ae}")
                                    speak("I had an issue with my speech recognition setup.")
                                    transcribed_text = "" # Ensure it's empty on error
                                except Exception as e_stt:
                                    print(f"Error during Whisper transcription: {e_stt}")
                                    speak("I had trouble processing your speech for transcription.")
                                    transcribed_text = "" # Ensure it's empty on error
                                # --- End Actual STT ---
                            except Exception as e_transcribe: # This outer try-except was for the simulated block, might need adjustment
                                print(f"Error during transcription block: {e_transcribe}") # Changed from "Error during transcription"
                                speak("I had trouble processing your speech.")
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
                # Try to match transcribed_text to AVAILABLE_WAKE_WORDS
                for i, ww_data in enumerate(AVAILABLE_WAKE_WORDS):
                    # Match by name (case-insensitive)
                    if ww_data['name'].lower() in transcribed_text:
                        chosen_index = i
                        break
                    # Match by number (e.g., "number one", "one", "1")
                    # Using re.search for more flexible number matching, e.g., "option 1", "number one"
                    if re.search(r'(number|option|#)\s*' + str(i + 1) + r'\b', transcribed_text, re.IGNORECASE) or \
                       transcribed_text == str(i + 1):
                        chosen_index = i
                        break
                if chosen_index != -1: break # Found a match by name or number pattern

            if chosen_index != -1: break # Break from voice_attempt loop
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
                if 1 <= choice <= len(AVAILABLE_WAKE_WORDS):
                    chosen_index = choice - 1
                    break
                else:
                    speak(f"That's not a valid number. Please choose between 1 and {len(AVAILABLE_WAKE_WORDS)}.")
            except ValueError:
                speak("That didn't seem like a number. Please try again.")

            if attempt < retries - 1:
                speak("Let's try that again.")
            else:
                speak("Sorry, I'm having trouble understanding your choice. Please try running the setup again later.")
                if 'tts_engine' in globals(): tts_engine.stop()
                sys.exit(1)

    selected_wake_word = AVAILABLE_WAKE_WORDS[chosen_index]
    speak(f"You've selected: {selected_wake_word['name']}. Great choice!")
    print(f"Selected wake word: {selected_wake_word['name']} (Model: {selected_wake_word['model_file']})")
    time.sleep(0.5)

    # 3. Verify model file existence (basic check)
    # Assumes script is run from project root where wakeword_models/ is located.
    model_path_to_check = selected_wake_word['model_file']
    if not os.path.exists(model_path_to_check):
        speak(f"Warning: The model file for {selected_wake_word['name']} at {model_path_to_check} was not found.")
        print(f"WARNING: Model file '{model_path_to_check}' not found!")
        print("Please ensure you have downloaded the .ppn file from Picovoice Console and placed it correctly.")
        speak("Please check the file path and run the setup again.")
        if 'tts_engine' in globals(): tts_engine.stop()
        sys.exit(1)
    else:
        print(f"Model file found at: {model_path_to_check}")


    # 4. Save Configuration
    speak("Saving your configuration...")
    config_data = DEFAULT_CONFIG.copy() # Start with defaults
    config_data["first_run_complete"] = True
    config_data["chosen_wake_word_engine"] = "picovoice"
    config_data["chosen_wake_word_model_path"] = selected_wake_word["model_file"]
    # We don't store the key itself, just confirm it was set in env at time of setup
    config_data["picovoice_access_key_is_set_env"] = True

    if save_config(config_data):
        speak("Configuration saved successfully.")
        print("Configuration saved.")
    else:
        speak("There was an error saving your configuration. Please try again.")
        print("ERROR: Failed to save configuration.")
        if 'tts_engine' in globals(): tts_engine.stop()
        sys.exit(1)

    time.sleep(0.5)
    speak("Setup is complete! Please restart the main application for the changes to take effect.")
    print("\nSetup complete. Please restart the main application.")

    # Gracefully stop TTS engine
    if 'tts_engine' in globals():
        tts_engine.stop()

if __name__ == "__main__":
    # Create dummy wakeword_models directory and files for standalone testing if they don't exist
    # In actual use, the developer provides these.
    if not os.path.exists("wakeword_models"):
        os.makedirs("wakeword_models")
        print("Created dummy 'wakeword_models/' directory for testing.")
    for ww in AVAILABLE_WAKE_WORDS:
        if not os.path.exists(ww["model_file"]):
            try:
                with open(ww["model_file"], "w") as f: # Create dummy files
                    f.write("This is a dummy Picovoice model file.\n")
                print(f"Created dummy model file: {ww['model_file']}")
            except Exception as e_file:
                print(f"Could not create dummy model file {ww['model_file']}: {e_file}")
                print("Please ensure the 'wakeword_models' directory is writable or create the files manually for testing.")

    # For standalone testing, ensure PICOVOICE_ACCESS_KEY is set, or mock it.
    if not os.environ.get("PICOVOICE_ACCESS_KEY"):
        print("WARNING: PICOVOICE_ACCESS_KEY environment variable not set for testing.")
        print("The script might exit if it requires it. For full test, please set it.")
        # os.environ["PICOVOICE_ACCESS_KEY"] = "TEST_KEY_ONLY_FOR_LOCAL_RUN" # Uncomment to mock for test

    run_first_time_setup()
