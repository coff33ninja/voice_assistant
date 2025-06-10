import json
import os
import logging
from typing import Dict, Any

# Use Windows AppData for config location
CONFIG_DIR = os.path.join(os.getenv('APPDATA', os.path.expanduser('~')), 'voice_assistant')
CONFIG_FILE_PATH = os.path.join(CONFIG_DIR, 'user_settings.json')
BACKUP_FILE_PATH = os.path.join(CONFIG_DIR, 'user_settings_backup.json')

DEFAULT_CONFIG: Dict[str, Any] = {
    "first_run_complete": False,
    "chosen_wake_word_engine": None,
    "chosen_wake_word_model_path": None,
    "picovoice_access_key_is_set_env": False,
    "chosen_tts_voice_id": None,
    "api_keys": {},  # For weather, etc.
    "language": "en"  # User language preference
}

def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure all required config keys are present, filling in defaults as needed.

    Args:
        config (dict): The config dictionary to validate.

    Returns:
        dict: The validated config dictionary.
    """
    for key in DEFAULT_CONFIG.keys():
        if key not in config:
            logging.warning(f"Missing key '{key}' in config. Using default.")
            config[key] = DEFAULT_CONFIG[key]
    return config

def load_config() -> Dict[str, Any]:
    """
    Loads the user configuration from CONFIG_FILE_PATH.

    Returns a dictionary with configuration data.
    Returns DEFAULT_CONFIG if the file doesn't exist or is invalid.
    Also auto-saves default config if missing.

    Returns:
        dict: The loaded or default config.
    """
    if not os.path.exists(CONFIG_FILE_PATH):
        save_config(DEFAULT_CONFIG)
        return DEFAULT_CONFIG.copy()
    try:
        with open(CONFIG_FILE_PATH, "r") as f:
            config_data = json.load(f)
            merged_config = DEFAULT_CONFIG.copy()
            merged_config.update(config_data)
            return validate_config(merged_config)
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON from {CONFIG_FILE_PATH}. File might be corrupted. Returning default config.")
        save_config(DEFAULT_CONFIG)
        return DEFAULT_CONFIG.copy()
    except Exception as e:
        logging.error(f"An unexpected error occurred while loading {CONFIG_FILE_PATH}: {e}. Returning default config.")
        save_config(DEFAULT_CONFIG)
        return DEFAULT_CONFIG.copy()

def save_config(data: Dict[str, Any]) -> bool:
    """
    Saves the given data dictionary to the configuration file.
    Creates a backup before overwriting.

    Args:
        data (dict): The configuration data to save.

    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        os.makedirs(CONFIG_DIR, exist_ok=True)
        # Backup before overwrite
        if os.path.exists(CONFIG_FILE_PATH):
            try:
                with open(CONFIG_FILE_PATH, "r") as orig, open(BACKUP_FILE_PATH, "w") as backup:
                    backup.write(orig.read())
                logging.info(f"Backup created at {BACKUP_FILE_PATH}")
            except Exception as e:
                logging.warning(f"Could not create backup: {e}")
        with open(CONFIG_FILE_PATH, "w") as f:
            json.dump(data, f, indent=4)
        logging.info(f"Configuration saved to {CONFIG_FILE_PATH}")
        return True
    except (IOError, PermissionError, json.JSONDecodeError) as e:
        logging.error(f"Failed to save config: {e}")
        return False

def get_config_path() -> str:
    """
    Returns the path to the user configuration file.

    Returns:
        str: The config file path.
    """
    return CONFIG_FILE_PATH

if __name__ == '__main__':
    """
    Example usage and test for user_config.py.
    """
    logging.basicConfig(level=logging.INFO)

    # Test loading (might be default if file doesn't exist)
    current_settings = load_config()
    logging.info(f"Loaded settings: {current_settings}")

    # Modify and save
    current_settings["first_run_complete"] = True
    current_settings["chosen_wake_word_engine"] = "picovoice_test"
    if save_config(current_settings):
        logging.info("Settings updated and saved.")

    # Test reloading
    reloaded_settings = load_config()
    logging.info(f"Reloaded settings: {reloaded_settings}")

    # Clean up test file
    # if os.path.exists(CONFIG_FILE_PATH):
    #     os.remove(CONFIG_FILE_PATH)
    #     logging.info(f"Cleaned up test file: {CONFIG_FILE_PATH}")
