import os
import sys
import urllib.request
import subprocess
from dotenv import set_key # For .env manipulation
from .config import (
    _PROJECT_ROOT, # Import _PROJECT_ROOT
    TTS_MODEL_NAME as CURRENT_EFFECTIVE_TTS_MODEL,
    TTS_SPEED_RATE as CURRENT_EFFECTIVE_TTS_SPEED,
    TTS_SAMPLERATE,
    DEFAULT_SPEAKER_WAV_PATH
)
from modules.stt_model_selection import prompt_stt_model_choice

def download_file(url, dest):
    try:
        urllib.request.urlretrieve(url, dest)
        print(f"Downloaded {dest}")
    except Exception as e:
        print(f"Failed to download {url}: {e}")

def play_sample_tts(model_name_for_tts: str, speed_rate: float, sample_text: str = "This is a test of the current speech rate."):
    try:
        from TTS.api import TTS as CoquiTTS # Import CoquiTTS locally
        # Import sounddevice locally to avoid issues if it's not available during other setup steps
        # or if this function is called in a context where sd is not globally defined.
        import sounddevice as sd
        import torch # Add import

        # --- Add this block to handle PyTorch 2.6+ weights_only=True issue for XTTS ---
        if "xtts" in model_name_for_tts.lower():
            print("Sample TTS: XTTS model detected, attempting to add safe globals for PyTorch 2.6+ compatibility.")
            safe_globals_to_add = []
            # Attempt to import and add XttsConfig
            try:
                from TTS.tts.configs.xtts_config import XttsConfig
                safe_globals_to_add.append(XttsConfig)
                print("Sample TTS: Identified TTS.tts.configs.xtts_config.XttsConfig for safe globals.")
            except ImportError:
                print("Sample TTS Warning: Could not import TTS.tts.configs.xtts_config.XttsConfig.")
            except Exception as e:
                print(f"Sample TTS Warning: Error importing XttsConfig: {e}")

            # Attempt to import and add XttsAudioConfig (based on error messages)
            try:
                from TTS.tts.models.xtts import XttsAudioConfig # Corrected import path based on typical structure
                safe_globals_to_add.append(XttsAudioConfig)
                print("Sample TTS: Identified TTS.tts.models.xtts.XttsAudioConfig for safe globals.")
            except ImportError:
                print("Sample TTS Warning: Could not import TTS.tts.models.xtts.XttsAudioConfig.")
            except Exception as e:
                print(f"Sample TTS Warning: Error importing XttsAudioConfig: {e}")

            # Attempt to import and add BaseDatasetConfig
            try:
                from TTS.config.shared_configs import BaseDatasetConfig
                safe_globals_to_add.append(BaseDatasetConfig)
                print("Sample TTS: Identified TTS.config.shared_configs.BaseDatasetConfig for safe globals.")
            except ImportError:
                print("Sample TTS Warning: Could not import TTS.config.shared_configs.BaseDatasetConfig.")
            except Exception as e:
                print(f"Sample TTS Warning: Error importing BaseDatasetConfig: {e}")
            
            # Attempt to import and add XttsArgs
            try:
                from TTS.tts.models.xtts import XttsArgs
                safe_globals_to_add.append(XttsArgs)
                print("Sample TTS: Identified TTS.tts.models.xtts.XttsArgs for safe globals.")
            except ImportError:
                print("Sample TTS Warning: Could not import TTS.tts.models.xtts.XttsArgs.")
            except Exception as e:
                print(f"Sample TTS Warning: Error importing XttsArgs: {e}")
            if safe_globals_to_add:
                try:
                    if hasattr(torch.serialization, 'add_safe_globals'):
                        torch.serialization.add_safe_globals(safe_globals_to_add)
                        print(f"Sample TTS: Applied {len(safe_globals_to_add)} identified XTTS class(es) to torch safe globals.")
                    else:
                        print("Sample TTS Warning: torch.serialization.add_safe_globals not found. This is expected for PyTorch < 2.1. XTTS models might still fail with PyTorch 2.6+ if this utility is missing.")
                except Exception as e: # Catch any exception during the add_safe_globals call
                    print(f"Sample TTS Error: Could not apply safe globals: {e}")
        # --- End block ---

        print(f"Attempting to generate sample audio with model '{model_name_for_tts}' at speed {speed_rate}...")
        tts_temp_instance = CoquiTTS(model_name=model_name_for_tts, progress_bar=False) # progress_bar=False for quick init

        tts_kwargs = {}
        if "xtts" in model_name_for_tts.lower():
            print(f"Model '{model_name_for_tts}' detected as an XTTS model. Checking for speaker WAV and language.")
            tts_kwargs["language"] = "en" # Default language for XTTS sample
            if os.path.exists(DEFAULT_SPEAKER_WAV_PATH):
                tts_kwargs["speaker_wav"] = DEFAULT_SPEAKER_WAV_PATH
                print(f"Using default speaker WAV: {DEFAULT_SPEAKER_WAV_PATH} and language: {tts_kwargs['language']}")
            else:
                print(f"WARNING: Default speaker WAV for XTTS models not found at '{DEFAULT_SPEAKER_WAV_PATH}'.")
                print("Cannot play TTS sample for this XTTS model without a speaker_wav.")
                print("You can still set the speed, but it won't be audibly tested now.")
                print(f"Please create a short .wav file (e.g., 3-5 seconds of speech) at '{DEFAULT_SPEAKER_WAV_PATH}' or choose a non-XTTS model for sample playback.")
                return False # Indicate sample playback failure

        print(f"Generating audio with: text='{sample_text}', speed={speed_rate}, model_specific_args={tts_kwargs}")
        audio_output = tts_temp_instance.tts(text=sample_text, speed=speed_rate, **tts_kwargs)

        # Determine the correct samplerate for playback. Some models might have their own.
        # For XTTS, the output samplerate is fixed by the model (e.g. 24000 Hz for xtts_v2).
        playback_samplerate = tts_temp_instance.synthesizer.output_sample_rate if hasattr(tts_temp_instance, 'synthesizer') and tts_temp_instance.synthesizer is not None and hasattr(tts_temp_instance.synthesizer, 'output_sample_rate') else TTS_SAMPLERATE
        sd.play(audio_output, samplerate=playback_samplerate)
        sd.wait()
        return True # Indicate success
    except Exception as e:
        print(f"Error playing TTS sample: {e}. Please ensure TTS and audio playback are working.")
        return False # Indicate failure

def setup_tts():
    from TTS.api import TTS as CoquiTTS # Import CoquiTTS locally for final initialization
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

    # --- TTS Speed Configuration ---
    print("\n--- TTS Speed Configuration ---")
    print("You can adjust the speaking speed of the TTS voice.")
    print("A value of 1.0 is normal speed. Values less than 1.0 are slower, greater than 1.0 are faster.")
    print("Note: Not all voice models support speed adjustment, or they may respond differently. Errors can occur with incompatible models/speeds.")
    print("E.g., 0.8 for slower, 1.2 for faster. The effect may vary by model.")

    while True:
        # Use final_model_to_use for the sample playback
        current_speed_for_prompt = float(os.getenv("TTS_SPEED_RATE", CURRENT_EFFECTIVE_TTS_SPEED)) # Get fresh value from .env or default
        print(f"\nCurrent effective TTS speed rate: {current_speed_for_prompt}")

        user_input_speed_rate_str = input(f"Enter new TTS speed rate (e.g., 1.0, or press Enter to test current '{current_speed_for_prompt}'): ").strip()

        speed_to_test = current_speed_for_prompt

        if not user_input_speed_rate_str: # User pressed Enter, test current
            print(f"Testing current speed: {speed_to_test}")
        else:
            try:
                speed_to_test = float(user_input_speed_rate_str)
            except ValueError:
                print("Invalid input. Please enter a valid number (e.g., 0.9, 1.0, 1.1).")
                continue

        sample_played_successfully = play_sample_tts(final_model_to_use, speed_to_test)

        if sample_played_successfully:
            confirm_speed = input("Did the sample play correctly and are you happy with this speed? (yes/no): ").strip().lower()
            if confirm_speed in ['yes', 'y']:
                if set_key(env_path, "TTS_SPEED_RATE", str(speed_to_test), quote_mode="always"):
                    print(f"TTS speed rate set to '{speed_to_test}' in {env_path}")
                else:
                    print(f"Warning: Could not save TTS_SPEED_RATE to {env_path}. Using '{speed_to_test}' for this session.")
                break # Exit the speed configuration loop
            else:
                print("Let's try a different speed or re-test.")
        else:
            error_choice = input("There was an error playing the sample. (c)ontinue trying different speeds, or (s)kip speed adjustment for this model? ").strip().lower()
            if error_choice == 's':
                print("Skipping TTS speed adjustment for this session. The default or previously set speed will be used.")
                # To ensure a default is in .env if nothing was set before and user skips:
                if not os.getenv("TTS_SPEED_RATE"): # Check if it's not already in .env
                    set_key(env_path, "TTS_SPEED_RATE", str(CURRENT_EFFECTIVE_TTS_SPEED), quote_mode="always")
                break # Exit the speed configuration loop
            # If 'c' or anything else, the loop continues to allow trying another speed.
            # Loop continues

    try:
        import torch # Add import
        # --- Add this block to handle PyTorch 2.6+ weights_only=True issue for XTTS ---
        if "xtts" in final_model_to_use.lower():
            print("TTS Setup: XTTS model detected, attempting to add safe globals for PyTorch 2.6+ compatibility.")
            safe_globals_to_add = []
            # Attempt to import and add XttsConfig
            try:
                from TTS.tts.configs.xtts_config import XttsConfig
                safe_globals_to_add.append(XttsConfig)
                print("TTS Setup: Identified TTS.tts.configs.xtts_config.XttsConfig for safe globals.")
            except ImportError:
                print("TTS Setup Warning: Could not import TTS.tts.configs.xtts_config.XttsConfig.")
            except Exception as e:
                print(f"TTS Setup Warning: Error importing XttsConfig: {e}")

            # Attempt to import and add XttsAudioConfig (based on error messages)
            try:
                from TTS.tts.models.xtts import XttsAudioConfig # Corrected import path based on typical structure
                safe_globals_to_add.append(XttsAudioConfig)
                print("TTS Setup: Identified TTS.tts.models.xtts.XttsAudioConfig for safe globals.")
            except ImportError:
                print("TTS Setup Warning: Could not import TTS.tts.models.xtts.XttsAudioConfig.")
            except Exception as e:
                print(f"TTS Setup Warning: Error importing XttsAudioConfig: {e}")

            # Attempt to import and add BaseDatasetConfig
            try:
                from TTS.config.shared_configs import BaseDatasetConfig
                safe_globals_to_add.append(BaseDatasetConfig)
                print("TTS Setup: Identified TTS.config.shared_configs.BaseDatasetConfig for safe globals.")
            except ImportError:
                print("TTS Setup Warning: Could not import TTS.config.shared_configs.BaseDatasetConfig.")
            except Exception as e:
                print(f"TTS Setup Warning: Error importing BaseDatasetConfig: {e}")

            # Attempt to import and add XttsArgs
            try:
                from TTS.tts.models.xtts import XttsArgs
                safe_globals_to_add.append(XttsArgs)
                print("TTS Setup: Identified TTS.tts.models.xtts.XttsArgs for safe globals.")
            except ImportError:
                print("TTS Setup Warning: Could not import TTS.tts.models.xtts.XttsArgs.")
            except Exception as e:
                print(f"TTS Setup Warning: Error importing XttsArgs: {e}")
            if safe_globals_to_add:
                try:
                    if hasattr(torch.serialization, 'add_safe_globals'):
                        torch.serialization.add_safe_globals(safe_globals_to_add)
                        print(f"TTS Setup: Applied {len(safe_globals_to_add)} identified XTTS class(es) to torch safe globals.")
                    else:
                        print("TTS Setup Warning: torch.serialization.add_safe_globals not found. This is expected for PyTorch < 2.1. XTTS models might still fail with PyTorch 2.6+ if this utility is missing.")
                except Exception as e: # Catch any exception during the add_safe_globals call
                    print(f"TTS Setup Error: Could not apply safe globals: {e}")
        # --- End block ---
        print(f"Initializing Coqui TTS with model '{final_model_to_use}' to download (if needed)...")
        CoquiTTS(model_name=final_model_to_use, progress_bar=True) # This is where the error occurred
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

def setup_stt_model():
    """Interactive setup for STT model selection (WhisperX or Whisper)."""
    choice = prompt_stt_model_choice()
    env_path = os.path.join(_PROJECT_ROOT, ".env")
    set_key(env_path, "STT_BACKEND", choice)
    print(f"STT backend set to '{choice}' in .env.")
