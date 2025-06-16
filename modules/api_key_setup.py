import os
import getpass
from typing import Optional

def setup_api_key(key_file_path: str, service_name: str, prompt_message: Optional[str] = None):
    """
    Prompts the user for an API key and saves it to the specified file.
    """
    if not os.path.exists(key_file_path):
        if prompt_message is None:
            prompt_message = f"Enter {service_name} Access Key (or press Enter to skip): "
        key = getpass.getpass(prompt_message)
        if key:
            os.makedirs(os.path.dirname(key_file_path), exist_ok=True)
            with open(key_file_path, "w") as f:
                f.write(key)
            print(f"{service_name} key saved to {key_file_path}.")
        else:
            print(f"Skipped {service_name} key setup. The service may not function.")
