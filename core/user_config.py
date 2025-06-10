import json
import os
import logging

# For simplicity, config is stored in the current working directory.
# For a real application, consider user's config directory:
# os.path.join(os.path.expanduser("~"), ".config", "voice_assistant", "config.json")
CONFIG_FILE_PATH = "user_settings.json"

DEFAULT_CONFIG = {
    "first_run_complete": False,
    "chosen_wake_word_engine": None, # e.g., "openwakeword" or "picovoice"
    "chosen_wake_word_model_path": None, # Path to .onnx for openwakeword or .ppn for picovoice
    "picovoice_access_key_is_set_env": False # Tracks if user confirmed env var is set
    # Add other future user-specific settings here
}

def load_config():
    """Loads the user configuration from CONFIG_FILE_PATH.
    Returns a dictionary with configuration data.
    Returns DEFAULT_CONFIG if the file doesn't exist or is invalid.
    """
    if not os.path.exists(CONFIG_FILE_PATH):
        logging.info(f"Configuration file not found at {CONFIG_FILE_PATH}. Returning default config.")
        # Optionally, save default config on first load if not found
        # save_config(DEFAULT_CONFIG)
        return DEFAULT_CONFIG.copy() # Return a copy to avoid modifying the global default

    try:
        with open(CONFIG_FILE_PATH, "r") as f:
            config_data = json.load(f)
            # Merge with defaults to ensure all keys are present
            # Loaded config takes precedence
            merged_config = DEFAULT_CONFIG.copy()
            merged_config.update(config_data)
            return merged_config
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON from {CONFIG_FILE_PATH}. File might be corrupted. Returning default config.")
        return DEFAULT_CONFIG.copy()
    except Exception as e:
        logging.error(f"An unexpected error occurred while loading {CONFIG_FILE_PATH}: {e}. Returning default config.")
        return DEFAULT_CONFIG.copy()

def save_config(data):
    """Saves the given data dictionary to the configuration file.
    Args:
        data (dict): The configuration data to save.
    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        # Ensure the directory exists if a more complex path is used in future
        # config_dir = os.path.dirname(CONFIG_FILE_PATH)
        # if config_dir and not os.path.exists(config_dir):
        # os.makedirs(config_dir)

        with open(CONFIG_FILE_PATH, "w") as f:
            json.dump(data, f, indent=4)
        logging.info(f"Configuration saved to {CONFIG_FILE_PATH}")
        return True
    except IOError as e:
        logging.error(f"Error writing configuration to {CONFIG_FILE_PATH}: {e}")
        return False
    except Exception as e:
        logging.error(f"An unexpected error occurred while saving {CONFIG_FILE_PATH}: {e}")
        return False

if __name__ == '__main__':
    # Example usage and test
    logging.basicConfig(level=logging.INFO)

    # Test loading (might be default if file doesn't exist)
    current_settings = load_config()
    print(f"Loaded settings: {current_settings}")

    # Modify and save
    current_settings["first_run_complete"] = True
    current_settings["chosen_wake_word_engine"] = "picovoice_test"
    if save_config(current_settings):
        print("Settings updated and saved.")

    # Test reloading
    reloaded_settings = load_config()
    print(f"Reloaded settings: {reloaded_settings}")

    # Clean up test file
    # if os.path.exists(CONFIG_FILE_PATH):
    #     os.remove(CONFIG_FILE_PATH)
    #     print(f"Cleaned up test file: {CONFIG_FILE_PATH}")
