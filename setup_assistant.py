import os
import json
import subprocess
import sys
from modules.install_dependencies import install_dependencies
from dotenv import load_dotenv # Import load_dotenv
from modules.download_and_models import setup_tts, setup_precise
from modules.api_key_setup import setup_api_key
from modules.whisperx_setup import setup_whisperx
from modules.db_setup import setup_db
from modules.utils import create_directories
from modules.config import (
    DB_PATH,
    INTENT_DATASET_CSV,
    INTENT_MODEL_SAVE_PATH,
    PICOVOICE_KEY_FILE_PATH,
    OPENWEATHER_API_KEY_FILE_PATH,
    BASE_DIR, # Centralized models base directory,
    DEFAULT_SPEAKER_WAV_PATH # Centralized default speaker WAV path
)
# For final TTS message
from TTS.api import TTS as CoquiTTS
import sounddevice as sd

PRECISE_MODEL_URL = "https://github.com/MycroftAI/mycroft-precise/releases/download/v0.3.0/precise-engine_0.3.0_x86_64.tar.gz"
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

def yes_no_prompt(prompt_text: str) -> bool:
    while True:
        response = input(f"{prompt_text} (yes/no): ").strip().lower()
        if response in ["yes", "y"]:
            return True
        if response in ["no", "n"]:
            return False
        print("Invalid input. Please enter 'yes' or 'no'.")

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
    create_directories(BASE_DIR, INTENT_MODEL_SAVE_PATH) # Use centralized path

    # Import functions here to ensure they are available for the action_map
    from modules.dataset import create_dataset
    from modules.model_training import fine_tune_model

    action_map = {
        "dependencies": (install_dependencies, {}),
        "tts": (setup_tts, {}),
        "precise": (setup_precise, {"base_dir": BASE_DIR, "model_url": PRECISE_MODEL_URL}),
        "picovoice_api_key": (setup_api_key, {"key_file_path": PICOVOICE_KEY_FILE_PATH, "service_name": "Picovoice", "prompt_message": "Enter Picovoice Access Key (or press Enter to skip): "}),
        "openweather_api_key": (setup_api_key, {"key_file_path": OPENWEATHER_API_KEY_FILE_PATH, "service_name": "OpenWeather", "prompt_message": "Enter OpenWeather API Key (or press Enter to skip): "}),
        "whisperx": (setup_whisperx, {}),
        "db": (setup_db, {"DB_PATH": DB_PATH}),
        "dataset": (create_dataset, {"dataset_path": INTENT_DATASET_CSV}), # Use centralized path
        "model_training": (fine_tune_model, {"dataset_path": INTENT_DATASET_CSV, "model_save_path": INTENT_MODEL_SAVE_PATH}) # Use centralized paths
    }

    while True: # Main loop for the entire setup process
        checkpoints = load_checkpoints()

        for step_name in SETUP_STEPS:
            print(f"\n--- Processing Step: {step_name.replace('_', ' ').title()} ---")
            func_to_call, func_args = action_map[step_name]

            is_step_complete = checkpoints.get(step_name, False)

            # TTS is always interactive for model selection, so we run it and mark it complete for this pass.
            if step_name == "tts":
                try:
                    func_to_call(**func_args)
                    checkpoints[step_name] = True
                except Exception as e:
                    print(f"Error during TTS configuration: {e}")
                    checkpoints[step_name] = False # Mark as failed
                finally:
                    save_checkpoints(checkpoints)
                continue # Move to the next step in SETUP_STEPS

            # For other steps:
            run_this_step = False
            if not is_step_complete:
                print("This step is not yet marked as complete.")
                run_this_step = True
            else: # Step is marked complete
                if yes_no_prompt("This step is marked as complete. Do you want to redo it?"):
                    run_this_step = True
                else:
                    print("Skipping step.")

            if run_this_step:
                while True: # Loop for retrying a failed step
                    try:
                        print(f"Running setup for {step_name.replace('_', ' ').title()}...")
                        func_to_call(**func_args)
                        checkpoints[step_name] = True
                        print(f"Step '{step_name.replace('_', ' ').title()}' completed successfully.")
                        break # Exit retry loop for this step
                    except Exception as e:
                        print(f"Error during step '{step_name.replace('_', ' ').title()}': {e}")
                        checkpoints[step_name] = False

                        retry_choice = input("Step failed. Choose action: (r)etry, (s)kip this step for now, (e)xit setup: ").strip().lower()
                        if retry_choice == 'r':
                            print("Retrying step...")
                            # Continue in the retry loop
                        elif retry_choice == 's':
                            print("Skipping step for this pass.")
                            break # Exit retry loop, move to next step in SETUP_STEPS
                        elif retry_choice == 'e':
                            print("Exiting setup process.")
                            save_checkpoints(checkpoints)
                            return # Exit main function
                        else:
                            print("Invalid choice. Skipping step for this pass.")
                            break # Default to skip
                    finally:
                        save_checkpoints(checkpoints)

        # After iterating through all steps in SETUP_STEPS for one pass
        print("\n--- Current Setup Pass Complete ---")

        all_done_this_pass = all(checkpoints.get(s, False) for s in SETUP_STEPS)
        if all_done_this_pass:
            print("All setup steps are currently marked as complete.")
            if not yes_no_prompt("Do you want to review or redo any steps? (If no, setup will finish)"):
                break # Exit main while loop, setup is finished
        else:
            print("Some setup steps are not yet marked as complete or may have been skipped.")
            if not yes_no_prompt("Do you want to go through the setup steps again (incomplete/skipped steps will be attempted, completed steps will offer a redo option)? (If no, setup will finish with current state)"):
                break # Exit main while loop

        # If continuing, ask about resetting all non-TTS steps
        if yes_no_prompt("Do you want to mark ALL non-TTS steps as incomplete and start their setup over in the next pass?"):
            for step_name_to_reset in SETUP_STEPS:
                if step_name_to_reset != "tts":
                    checkpoints[step_name_to_reset] = False
            save_checkpoints(checkpoints)
            print("All non-TTS steps have been reset. Starting setup pass again.")
        # The main while loop will continue for another pass if not explicitly broken.

    print("\nSetup process finished!")
    print("Well, this is the most you are supposed to do for voice assistants, who knows maybe there might be new typing areas added later.")

    final_tts_message = "But let's continue, and you can finally hear my voice! I hope I sound just right after all that tinkering."
    print(final_tts_message)

    try:
        print("Playing final message...")
        # Reload .env to get the latest settings potentially changed by setup_tts()
        load_dotenv(override=True)
        
        # Fetch the potentially updated TTS configuration
        # Use defaults from modules.config if not found in .env (though setup_tts should ensure they are set)
        from modules.config import TTS_MODEL_NAME as DEFAULT_TTS_MODEL, TTS_SPEED_RATE as DEFAULT_TTS_SPEED, TTS_SAMPLERATE as DEFAULT_TTS_SAMPLERATE
        
        current_tts_model = os.getenv("TTS_MODEL_NAME", DEFAULT_TTS_MODEL)
        current_tts_speed = float(os.getenv("TTS_SPEED_RATE", str(DEFAULT_TTS_SPEED)))
        current_tts_samplerate = int(os.getenv("TTS_SAMPLERATE", str(DEFAULT_TTS_SAMPLERATE))) # Samplerate is usually fixed but good to be consistent
        
        tts_instance_final = CoquiTTS(model_name=current_tts_model, progress_bar=False)

        # --- Add these debug prints ---
        print(f"DEBUG (Final TTS): current_tts_model = '{current_tts_model}'")
        print(f"DEBUG (Final TTS): current_tts_model.lower() = '{current_tts_model.lower()}'")
        print(f"DEBUG (Final TTS): 'xtts' in current_tts_model.lower() = {'xtts' in current_tts_model.lower()}")

        tts_kwargs_final = {}
        if "xtts" in current_tts_model.lower():
            print(f"Final TTS: Model '{current_tts_model}' detected as an XTTS model. Checking for speaker WAV.")
            tts_kwargs_final["language"] = "en" # Default language for XTTS
            if os.path.exists(DEFAULT_SPEAKER_WAV_PATH):
                tts_kwargs_final["speaker_wav"] = DEFAULT_SPEAKER_WAV_PATH
                print(f"Final TTS: Using default speaker WAV: {DEFAULT_SPEAKER_WAV_PATH} and language: {tts_kwargs_final['language']}")
                print(f"DEBUG (Final TTS): DEFAULT_SPEAKER_WAV_PATH exists: True")
            else:
                print(f"WARNING (Final TTS): Default speaker WAV for XTTS models not found at '{DEFAULT_SPEAKER_WAV_PATH}'.")
                print("The final message might not play correctly or use a default voice.")
                # Allow to proceed, it might use a default internal speaker if any, or fail as observed.

        audio_output = tts_instance_final.tts(text=final_tts_message, speed=current_tts_speed, **tts_kwargs_final)
        
        # Determine the correct samplerate for playback for the final message
        final_playback_samplerate = current_tts_samplerate # Default
        if hasattr(tts_instance_final, 'synthesizer') and tts_instance_final.synthesizer is not None and hasattr(tts_instance_final.synthesizer, 'output_sample_rate'):
            final_playback_samplerate = tts_instance_final.synthesizer.output_sample_rate
            print(f"Final TTS: Using model's output samplerate: {final_playback_samplerate}")

        sd.play(audio_output, samplerate=final_playback_samplerate)
        sd.wait()
    except Exception as tts_e:
        print(f"Could not play final TTS message: {tts_e}")
    if yes_no_prompt("Would you like to launch the voice assistant now?"):
        print("Launching voice assistant...")
        try:
            subprocess.Popen([sys.executable, "voice_assistant.py"])
            sys.exit(0) # Exit the setup script successfully
        except Exception as e:
            print(f"Failed to launch voice_assistant.py: {e}")
            print("You can run it manually using: python voice_assistant.py")
    else:
        print("You can run the assistant later using: python voice_assistant.py")

if __name__ == "__main__":
    main()
