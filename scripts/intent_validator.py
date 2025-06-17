import pandas as pd
import os

def validate_intent_responses_coverage(dataset_path: str, responses_path: str) -> bool:
    """
    Validates that every unique intent in the dataset CSV has a corresponding
    entry in the responses CSV.

    Args:
        dataset_path (str): Path to the intent_dataset.csv file.
        responses_path (str): Path to the intent_responses.csv file.

    Returns:
        bool: True if all intents are covered, False otherwise.
    """
    all_covered = True
    try:
        dataset_df = pd.read_csv(dataset_path)
        responses_df = pd.read_csv(responses_path)
    except FileNotFoundError as e:
        print(f"Error: Could not find one or both CSV files: {e}")
        return False
    except pd.errors.EmptyDataError as e:
        print(f"Error: One or both CSV files are empty or malformed: {e}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred while reading CSV files: {e}")
        return False

    if 'label' not in dataset_df.columns:
        print(f"Error: 'label' column not found in {dataset_path}")
        return False
    if 'intent' not in responses_df.columns:
        print(f"Error: 'intent' column not found in {responses_path}")
        return False

    unique_intents_in_dataset = set(dataset_df['label'].unique())
    defined_intents_in_responses = set(responses_df['intent'].unique())

    missing_responses = unique_intents_in_dataset - defined_intents_in_responses
    if missing_responses:
        print("\nValidation FAILED: Missing responses for the following intents:")
        for intent in sorted(list(missing_responses)):
            print(f"  - Intent '{intent}' is in '{os.path.basename(dataset_path)}' but has no corresponding response in '{os.path.basename(responses_path)}'.")
        all_covered = False
    else:
        print(f"\nValidation PASSED: All {len(unique_intents_in_dataset)} unique intents in '{os.path.basename(dataset_path)}' have corresponding entries in '{os.path.basename(responses_path)}'.")

    # Optional: Check for responses defined but not in dataset (orphaned responses)
    orphaned_responses = defined_intents_in_responses - unique_intents_in_dataset
    if orphaned_responses:
        print("\nINFO: The following intents have responses defined in "
              f"'{os.path.basename(responses_path)}' but are not found in the "
              f"'{os.path.basename(dataset_path)}' (orphaned responses):")
        for intent in sorted(list(orphaned_responses)):
            print(f"  - Intent '{intent}'")
        # This is informational, so it doesn't make all_covered False unless desired.

    return all_covered

if __name__ == "__main__":
    # Assuming the script is run from the project root (e.g., e:\SCRIPTS\voice_assistant\)
    # or that the paths are adjusted accordingly.
    project_root = os.path.dirname(os.path.abspath(__file__)) # Or a fixed path
    
    # Construct paths relative to the project root where voice_assistant.py is located
    # This assumes 'models' is a subdirectory of the project root.
    dataset_csv_path = os.path.join(project_root, "models", "intent_dataset.csv")
    responses_csv_path = os.path.join(project_root, "models", "intent_responses.csv")

    print(f"Validating '{dataset_csv_path}' against '{responses_csv_path}'...")
    
    # Create dummy CSV files for testing if they don't exist
    # In a real scenario, these files would already exist.
    os.makedirs(os.path.join(project_root, "models"), exist_ok=True)

    if not os.path.exists(dataset_csv_path):
        print(f"Creating dummy '{dataset_csv_path}' for validation script testing.")
        dummy_dataset_data = {'text': ['hello', 'goodbye', 'weather today', 'what time is it'],
                              'label': ['greeting', 'goodbye', 'get_weather', 'get_time']}
        pd.DataFrame(dummy_dataset_data).to_csv(dataset_csv_path, index=False)

    if not os.path.exists(responses_csv_path):
        print(f"Creating dummy '{responses_csv_path}' for validation script testing.")
        dummy_responses_data = {'intent': ['greeting', 'goodbye', 'get_weather', 'general_query_fallback'],
                                'response': ['Hello there!', 'See you!', 'The weather is sunny.', 'Sorry, I did not understand.']}
        pd.DataFrame(dummy_responses_data).to_csv(responses_csv_path, index=False)
        
    validation_successful = validate_intent_responses_coverage(dataset_csv_path, responses_csv_path)

    if validation_successful:
        print("\nAll checks passed or only informational messages were shown.")
    else:
        print("\nOne or more critical validation checks failed.")

