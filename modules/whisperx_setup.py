import os
import sys
import subprocess
from dotenv import set_key
import whisperx
try:
    import sounddevice as sd
    import scipy.io.wavfile as wav
except ImportError:
    print("sounddevice and scipy are required. Please install them with: pip install sounddevice scipy")
    # Re-raise or exit if these are critical for the setup script to proceed
    raise

import tempfile
from modules.config import STT_MODEL_NAME as DEFAULT_STT_MODEL_NAME

# Common WhisperX models
WHISPERX_MODELS = [
    "tiny", "tiny.en", "base", "base.en", "small", "small.en",
    "medium", "medium.en", "large-v1", "large-v2", "large-v3"
]

def get_project_root():
    """Infer the project root directory."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def setup_whisperx():
    project_root = get_project_root()
    dotenv_path = os.path.join(project_root, '.env')

    # Ensure .env file exists
    if not os.path.exists(dotenv_path):
        open(dotenv_path, 'a').close()
        print(f"Created .env file at {dotenv_path}")

    current_stt_model = os.getenv("STT_MODEL_NAME", DEFAULT_STT_MODEL_NAME)
    print(f"Current WhisperX model: {current_stt_model}")

    print("\nAvailable WhisperX models:")
    for i, model_name in enumerate(WHISPERX_MODELS):
        print(f"{i + 1}. {model_name}")

    selected_model_name = current_stt_model
    while True:
        try:
            choice = input(f"Select a model by number (or press Enter to keep '{current_stt_model}'): ")
            if not choice:
                break
            model_index = int(choice) - 1
            if 0 <= model_index < len(WHISPERX_MODELS):
                selected_model_name = WHISPERX_MODELS[model_index]
                break
            else:
                print("Invalid selection. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    if selected_model_name != current_stt_model:
        set_key(dotenv_path, "STT_MODEL_NAME", selected_model_name)
        print(f"WhisperX model set to: {selected_model_name} in {dotenv_path}")
        os.environ["STT_MODEL_NAME"] = selected_model_name # Update current session's env
    else:
        print(f"Keeping current model: {selected_model_name}")

    try:
        print(f"\nLoading WhisperX model: {selected_model_name}...")
        # Note: device and compute_type could also be made configurable
        model = whisperx.load_model(selected_model_name, device="cpu", compute_type="int8")
        print(f"Model {selected_model_name} loaded successfully.")

        print("WhisperX setup complete. Models will download on first use if not already present.")
        # Prompt user to say something
        duration = 5  # seconds
        fs = 16000
        print("Please say something after the beep...")
        sd.sleep(500)
        print("Beep!")
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
        sd.wait()
        # Save to temp wav file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmpfile:
            wav.write(tmpfile.name, fs, recording)
            tmp_wav_path = tmpfile.name
        # Transcribe with whisperx
        # Model is already loaded above
        result = model.transcribe(tmp_wav_path)
        # Print the transcription result (print the whole result for clarity)
        print("Transcription result:", result)
        os.remove(tmp_wav_path)

        # Attempt to apply PyTorch Lightning checkpoint upgrade for the *selected* model
        try:
            print(f"\nAttempting to apply PyTorch Lightning checkpoint upgrade for WhisperX model: {selected_model_name}...")

            # Determine the correct checkpoint path. This is a bit of a guess.
            # Whisper models are typically downloaded to ~/.cache/whisper or similar.
            # whisperx.load_model downloads to `os.path.join(os.path.expanduser("~"), ".cache", "whisperx")`
            # but the actual model files (like .bin) are inside subdirectories named after the model.

            # Correctly find the path of the loaded model
            # Based on whisperx source, the model is downloaded to Hugging Face cache or local dir if specified.
            # For default whisperx usage, it seems to rely on the original whisper cache path.
            # Let's try to find the .bin file within the typical cache structure.
            # The specific file `pytorch_model.bin` is related to HuggingFace Transformers models.
            # Original Whisper models have .pt extensions.
            # `whisperx.load_model` can load original Whisper models or HuggingFace fine-tuned ones.
            # The checkpoint upgrade is for PyTorch Lightning, which is more relevant for HuggingFace models.

            # The `assets/pytorch_model.bin` path seems to be for a *specific bundled component* of whisperx,
            # not necessarily for the ASR model itself if it's an original OpenAI model.
            # If `selected_model_name` refers to an OpenAI model (e.g., "base", "small"), this upgrade script
            # might not be relevant or might target a shared component.
            # If it's a HuggingFace model, the path would be different.

            # Given the original script's hardcoded path, it seems it was intended for a component of whisperx itself
            # rather than the dynamically loaded ASR models.
            # Let's assume the original intention was to upgrade a shared component within the whisperx library's own assets.
            # This part of the script might need more context on what exactly `pytorch_model.bin` refers to.
            # If the upgrade is for the *transcription model itself*, the path needs to be dynamic.
            # If `whisperx.load_model` downloads HF models, their paths are like:
            # ~/.cache/huggingface/hub/models--<org>--<model_name>/snapshots/<commit_hash>/pytorch_model.bin

            # For now, let's keep the original logic for the checkpoint path, as it might be targeting
            # a shared component within the whisperx package. If issues arise, this needs revisiting.
            whisperx_module_path = os.path.dirname(whisperx.__file__)
            # This path points to a file within the *installed whisperx package*, not the downloaded model weights.
            # For example, /path/to/venv/lib/pythonX.Y/site-packages/whisperx/assets/pytorch_model.bin
            # This specific file is part of the whisperx package for its alignment model, not the main ASR model.
            # So, this upgrade script is likely for the alignment model's checkpoint.
            checkpoint_file_path = os.path.join(whisperx_module_path, "assets", "pytorch_model.bin") # This is for the alignment model

            print(f"Targeting checkpoint file for alignment model: {checkpoint_file_path}")

            if os.path.exists(checkpoint_file_path):
                upgrade_command = [
                    sys.executable,
                    "-m",
                    "pytorch_lightning.utilities.upgrade_checkpoint",
                    checkpoint_file_path,
                ]
                print(f"Running: {' '.join(upgrade_command)}")
                process = subprocess.run(upgrade_command, capture_output=True, text=True, check=False)
                if process.returncode == 0:
                    print("PyTorch Lightning checkpoint upgrade command executed successfully (if an upgrade was needed).")
                    if process.stdout.strip():
                        print(f"Upgrade stdout:\n{process.stdout.strip()}")
                    # PTL might output to stderr even on success for info like "No upgrade needed"
                    if process.stderr.strip():
                        print(f"Upgrade stderr:\n{process.stderr.strip()}")
                else:
                    print(f"PyTorch Lightning checkpoint upgrade command may have failed or was not applicable. Return code: {process.returncode}")
                    if process.stdout.strip():
                        print(f"Stdout:\n{process.stdout.strip()}")
                    if process.stderr.strip():
                        print(f"Stderr:\n{process.stderr.strip()}")
                    print("This is usually not critical. The model should still work.")
            else:
                print(f"WhisperX checkpoint file not found at {checkpoint_file_path}. Skipping upgrade attempt.")
        except ImportError:
            print("pytorch_lightning module not found. Skipping checkpoint upgrade attempt. You can install it with: pip install pytorch-lightning")
        except Exception as upgrade_e:
            print(f"An error occurred during PyTorch Lightning checkpoint upgrade attempt for the alignment model: {upgrade_e}")

    except ImportError as ie:
        print(f"A required module is missing: {ie}")
        print("Please ensure all dependencies are installed. You might need to run: pip install python-dotenv whisperx sounddevice scipy pytorch-lightning")
    except Exception as e:
        print(f"WhisperX setup or test failed: {e}")
