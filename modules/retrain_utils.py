import asyncio
import subprocess
import sys
import os

def parse_retrain_request(text: str) -> bool:
    triggers = [
        "retrain the model", "update my assistant", "retrain assistant",
        "retrain intent classifier", "retrain", "train model", "update model"
    ]
    return any(trigger in text.lower() for trigger in triggers)

async def trigger_model_retraining_async() -> tuple[bool, str]:
    """
    Triggers the model retraining script.
    Assumes voice_assistant.py is run from the project root, and
    model_training.py is in the 'modules' subdirectory.
    Returns a tuple (success: bool, message: str).
    """
    # Path to model_training.py relative to the project root
    script_path = os.path.join("modules", "model_training.py")
    command = [sys.executable, script_path]

    print(f"Executing retraining: {' '.join(command)}")
    try:
        result = await asyncio.to_thread(
            subprocess.run,
            command,
            capture_output=True, text=True, check=False # check=False to handle errors
        )
        if result.returncode == 0:
            return True, "Retraining complete. Your assistant is now up to date."
        else:
            error_msg = f"Retraining failed. Stderr: {result.stderr.strip()}"
            if result.stdout: error_msg += f" Stdout: {result.stdout.strip()}"
            return False, error_msg
    except FileNotFoundError:
        return False, f"Retraining script '{script_path}' not found. Ensure it's in the correct location."
    except Exception as e:
        return False, f"Retraining failed due to an unexpected error: {e}"