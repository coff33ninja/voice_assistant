import pandas as pd
import asyncio
import os
import sys
from modules.retrain_utils import trigger_model_retraining_async

# Ensure the parent directory is in sys.path for module imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

DATASET_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'models', 'intent_dataset.csv'))
RESPONSES_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'models', 'intent_responses.csv'))

def validate_intents():
    print(f"Validating '{DATASET_PATH}' against '{RESPONSES_PATH}'...")
    dataset = pd.read_csv(DATASET_PATH)
    responses = pd.read_csv(RESPONSES_PATH)

    dataset_intents = set(dataset['label'].unique())
    response_intents = set(responses['intent'].unique())

    missing_in_responses = dataset_intents - response_intents
    unused_in_dataset = response_intents - dataset_intents

    if missing_in_responses:
        print("\nValidation FAILED: Missing responses for the following intents:")
        for intent in missing_in_responses:
            print(f"  - Intent '{intent}' is in 'intent_dataset.csv' but has no corresponding response in 'intent_responses.csv'.")
        print("\nOne or more critical validation checks failed.")
        return False
    else:
        print("All dataset intents have a response.")

    if unused_in_dataset:
        print("\nINFO: The following intents have responses defined in 'intent_responses.csv' but are not found in the 'intent_dataset.csv' (orphaned responses):")
        for intent in unused_in_dataset:
            print(f"  - Intent '{intent}'")
    else:
        print("All response intents are referenced in the dataset.")
    return True

if __name__ == "__main__":
    valid = validate_intents()
    if valid:
        print("\nValidation passed. Starting retraining...")
        result = asyncio.run(trigger_model_retraining_async())
        print(result[1])
    else:
        print("\nValidation failed. Please fix the issues above before retraining.")
