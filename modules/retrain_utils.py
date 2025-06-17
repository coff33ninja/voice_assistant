import asyncio
import subprocess
import sys
import os

# Define project root and paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATASET_PATH = os.path.join(PROJECT_ROOT, "models", "intent_dataset.csv")
MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, "models", "fine_tuned_intent_classifier")
MODEL_TRAINING_SCRIPT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_training.py")

def parse_retrain_request(text: str) -> bool:
    triggers = [
        "retrain the model", "update my assistant", "retrain assistant",
        "retrain intent classifier", "retrain", "train model", "update model"
    ]
    return any(trigger in text.lower() for trigger in triggers)

async def trigger_model_retraining_async() -> tuple[bool, str]:
    """
    Triggers the model retraining script using absolute paths and arguments.
    Returns a tuple (success: bool, message: str).
    """
    command = [
        sys.executable,
        MODEL_TRAINING_SCRIPT_PATH,
        DATASET_PATH,
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