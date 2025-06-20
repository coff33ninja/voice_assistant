import asyncio
import subprocess
import sys
import os

# Define project root and paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATASET_PATH = os.path.join(PROJECT_ROOT, "intent_data", "intent_dataset.csv")
MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, "models", "fine_tuned_intent_classifier")
MODEL_TRAINING_SCRIPT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_training.py")
AUGMENT_SCRIPT_PATH = os.path.join(PROJECT_ROOT, "scripts", "augment_intent_dataset.py")
AUGMENTED_DATASET_PATH = os.path.join(PROJECT_ROOT, "intent_data", "intent_dataset_augmented.csv")

def parse_retrain_request(text: str) -> bool:
    triggers = [
        "retrain the model", "update my assistant", "retrain assistant",
        "retrain intent classifier", "retrain", "train model", "update model"
    ]
    return any(trigger in text.lower() for trigger in triggers)

async def trigger_model_retraining_async() -> tuple[bool, str]:
    """
    Triggers the dataset augmentation, then the model retraining script using absolute paths and arguments.
    Returns a tuple (success: bool, message: str).
    """
    # 1. Run augmentation script
    augment_command = [sys.executable, AUGMENT_SCRIPT_PATH]
    print(f"Running dataset augmentation: {' '.join(augment_command)}")
    try:
        augment_result = await asyncio.to_thread(
            subprocess.run,
            augment_command,
            capture_output=True, text=True, check=False
        )
        if augment_result.returncode != 0:
            print(f"Warning: Augmentation failed. Stderr: {augment_result.stderr.strip()}")
        else:
            print(f"Augmentation output: {augment_result.stdout.strip()}")
    except Exception as e:
        print(f"Warning: Exception during augmentation: {e}")

    # 2. Use augmented dataset if it exists, else fallback
    dataset_path = AUGMENTED_DATASET_PATH if os.path.isfile(AUGMENTED_DATASET_PATH) else DATASET_PATH
    command = [
        sys.executable,
        MODEL_TRAINING_SCRIPT_PATH,
        dataset_path,
        MODEL_SAVE_PATH
    ]
    print(f"Executing retraining with command: {' '.join(command)}")
    try:
        result = await asyncio.to_thread(
            subprocess.run,
            command,
            capture_output=True, text=True, check=False
        )
        if result.returncode == 0:
            return True, "Retraining complete. Your assistant is now up to date."
        else:
            error_msg = f"Retraining failed. Stderr: {result.stderr.strip()}"
            if result.stdout:
                error_msg += f" Stdout: {result.stdout.strip()}"
            return False, error_msg
    except FileNotFoundError:
        return False, f"Retraining failed: Python executable or model training script not found. Check paths.\nPython: {sys.executable}\nScript: {MODEL_TRAINING_SCRIPT_PATH}"
    except Exception as e:
        return False, f"Retraining failed due to an unexpected error: {e}"