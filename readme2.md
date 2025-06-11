# Voice Assistant

This is a Python-based voice assistant capable of understanding voice commands and performing various tasks.

## Project Structure

The project is organized as follows:

*   **`main.py`**: The main entry point for the application. It initializes the `Assistant` class, which loads configurations, intent modules, and the core voice processing engine.
*   **`first_run_setup.py`**: A script that guides the user through initial configuration on the first run, including wake word engine selection, model setup, and TTS voice choice.
*   **`run_tests.py`**: A script to discover and execute all automated tests located in the `tests/` directory.
*   **`core/`**: This directory houses the essential components of the voice assistant:
    *   `engine.py`: The `VoiceCore` class, responsible for wake word detection (supporting Picovoice Porcupine and OpenWakeWord) and speech-to-text (using Whisper).
    *   `tts.py`: The `TTSEngine` class, providing text-to-speech capabilities using `pyttsx3`, with support for queuing and voice selection.
    *   `user_config.py`: Manages loading and saving user-specific configurations (e.g., chosen wake word, TTS voice) to a JSON file in the user's AppData directory.
*   **`modules/`**: Contains individual Python files, each representing a "skill" or a set of related commands for the assistant. Each module typically includes a `register_intents()` function that maps voice commands (phrases) to Python functions.
    *   `configs/`: (Within `modules/`) This subdirectory is used by some modules (like `server.py` and `ping.py`) to store their specific configurations, such as `systems_config.json` for server details.
*   **`tests/`**: This directory contains unit tests for various components of the project, ensuring functionality and stability.
*   **`wakeword_models/`**: (Not explicitly listed but created by `first_run_setup.py`) This directory is intended to store wake word model files (e.g., `.ppn` for Picovoice, `.onnx` for OpenWakeWord).
*   **`.env`**: (Optional, loaded by `main.py` and `first_run_setup.py`) Used to store environment variables like `PICOVOICE_ACCESS_KEY`.

## Current Modules and Intents

The assistant currently supports the following modules and voice commands:

### 1. `find_devices.py`
    *   "find devices": Scans the local network for active devices using ARP.
    *   "scan network": Alias for "find devices".

### 2. `general.py`
    *   "hello": A simple greeting.
    *   "what time is it": Tells the current time.
    *   "tell me the time": Alias for "what time is it".
    *   "run self test": Executes the project's test suite.
    *   "run a self test": Alias for "run self test".

### 3. `ping.py`
    *   "ping <target>": Pings a specified device by name (from `systems_config.json`) or IP address.
    *   "check status of <target>": Alias for "ping <target>".
    *   "what is the status of <target>": Alias for "ping <target>".

### 4. `server.py`
    *   "start server <server_name>": Boots a server using Wake-on-LAN (details from `systems_config.json`) and optionally pings it.
    *   "stop server": Placeholder for server shutdown functionality.

### 5. `speedtest.py`
    *   "run speed test": Runs an internet speed test using the `speedtest-cli` library.
    *   "check internet speed": Alias for "run speed test".

### 6. `system_info.py`
    *   "system status": Provides a summary of CPU, memory, disk usage, and uptime.
    *   "tell me system status": Alias for "system status".
    *   "what's the system status": Alias for "system status".
    *   "cpu usage": Reports current CPU utilization.
    *   "what's the cpu usage": Alias for "cpu usage".
    *   "tell me cpu load": Reports system load average (more relevant for Linux/macOS).
    *   "memory usage": Reports current memory (RAM) usage.
    *   "what's the memory usage": Alias for "memory usage".
    *   "ram status": Alias for "memory usage".
    *   "disk space": Reports disk usage for the default drive/path.
    *   "disk space for <path>": Reports disk usage for the specified path.
    *   "how much disk space is left": Alias for "disk space" (uses default path).
    *   "storage status": Alias for "disk space" (uses default path).
    *   "system uptime": Reports how long the system has been running.
    *   "how long has the system been running": Alias for "system uptime".
    *   "system load": Alias for "tell me cpu load".
    *   "what's the system load": Alias for "tell me cpu load".
    *   "what is the system load average": Alias for "tell me cpu load".

### 7. `weather.py`
    *   "get weather <city_name>": Fetches and speaks the current weather for the specified city using the Open-Meteo API.
    *   "weather in <city_name>": Alias for "get weather <city_name>".

### 8. `wol.py` (Wake-on-LAN)
    *   "wake on lan <mac_address>": Sends a Wake-on-LAN packet to the specified MAC address. (Note: `server.py` provides a more user-friendly interface using system names).
    *   "send wol packet <mac_address>": Alias for "wake on lan <mac_address>".

## Testing

The project includes a `tests/` directory with unit tests for various components. These can be run using `python run_tests.py`.

**Note:** Not all functions and modules are covered by automated tests yet.

## Setup

1.  Ensure Python 3.x is installed.
2.  Install dependencies (e.g., from a `requirements.txt` file if provided).
3.  (Optional) Create a `.env` file in the project root to store your `PICOVOICE_ACCESS_KEY` if you plan to use the Picovoice engine.
    ```
    PICOVOICE_ACCESS_KEY=your_picovoice_access_key_here
    ```
4.  Run `main.py`. On the first execution, `first_run_setup.py` will guide you through the necessary configurations.

## Usage

Once set up, run `main.py`. The assistant will listen for the configured wake word. After the wake word is detected, it will listen for a command.
You can then issue commands like "find devices", "what time is it", or any other supported command listed above.