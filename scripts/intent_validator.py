import pandas as pd
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from typing import Tuple
from modules.retrain_utils import trigger_model_retraining_async

# Ensure the parent directory is in sys.path for module imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

DATASET_PATH = os.path.abspath(
    os.path.join(SCRIPT_DIR, "..", "intent_data", "intent_dataset.csv")
)
RESPONSES_PATH = os.path.abspath(
    os.path.join(SCRIPT_DIR, "..", "intent_data", "intent_responses.csv")
)

def validate_intents():
    print(f"Validating '{DATASET_PATH}' against '{RESPONSES_PATH}'...")
    dataset = pd.read_csv(DATASET_PATH)
    responses = pd.read_csv(RESPONSES_PATH)

    dataset_intents = set(dataset['label'].unique())
    response_intents = set(responses['intent'].unique())

    validation_messages = []
    validation_passed = True

    missing_in_responses = dataset_intents - response_intents
    unused_in_dataset = response_intents - dataset_intents

    if missing_in_responses:
        msg = "\nValidation FAILED: Missing responses for the following intents:"
        print(msg)
        validation_messages.append(msg)
        validation_passed = False
        for intent in missing_in_responses:
            msg = f"  - Intent '{intent}' is in 'intent_dataset.csv' but has no corresponding response in 'intent_responses.csv'."
            print(msg)
            validation_messages.append(msg)
        validation_messages.append("\nOne or more critical validation checks failed.")
    else:
        print("All dataset intents have a response.")

    if unused_in_dataset:
        msg = "\nINFO: The following intents have responses defined in 'intent_responses.csv' but are not found in the 'intent_dataset.csv' (orphaned responses):"
        print(msg)
        # These are informational, so they don't cause validation_passed to be False
        # but we can still include them in messages if desired for a comprehensive report.
        # For now, just printing them is fine as per original logic.
        for intent in unused_in_dataset:
            print(f"  - Intent '{intent}'")
    else:
        print("All response intents are referenced in the dataset.")
    return validation_passed, validation_messages

async def run_validation_and_retrain_async() -> Tuple[bool, str]:
    print("Running intent validation before retraining...")
    validation_passed, validation_messages = validate_intents()
    if validation_passed:
        print("\nValidation passed. Starting retraining...")
        # Await the async function directly
        success, retrain_msg = await trigger_model_retraining_async()
        print(retrain_msg)
        return success, retrain_msg
    else:
        print("\nValidation failed. Please fix the issues above before retraining.")
        error_summary = "Validation failed. Model not retrained. Issues:\n" + "\n".join(validation_messages)
        return False, error_summary