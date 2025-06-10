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

    chosen_index = -1
    retries = 3
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
