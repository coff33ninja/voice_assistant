# Voice Assistant Project

This project is a Python-based voice assistant that listens for a wake word, processes voice commands, and performs actions based on recognized intents.

## Features

*   **Wake Word Detection**: Uses `openwakeword` or `Picovoice Porcupine` (configurable on first run) to listen for a specific wake word (e.g., "Hey Jimmy").
*   **Speech-to-Text**: Employs `whisper` (OpenAI) to transcribe voice commands into text.
*   **Text-to-Speech**: Utilizes `pyttsx3` for voice responses.
*   **Modular Intents**: Functionality is extended through modules located in the `modules` directory. Each module can define its own voice commands (intents) and corresponding actions.
*   **Core Engine**: A central `VoiceCore` manages audio input, wake word detection, and command processing.
*   **Asynchronous TTS**: Speech synthesis is handled in a separate thread to prevent blocking the main application.

## Project Structure

```
.
├── core/                 # Core components of the voice assistant
│   ├── __init__.py
│   ├── engine.py         # VoiceCore: wake word, STT
│   └── tts.py            # TTSEngine: text-to-speech
├── modules/              # Extensible modules for different commands
│   ├── __init__.py
│   ├── configs/          # Configuration files for modules
│   │   └── systems_config.json
│   ├── general.py        # Example module with general commands
│   └── ...               # Other module files (e.g., ping.py, wol.py)
├── tests/                # Unit and integration tests
│   ├── __init__.py
│   └── test_core.py      # Tests for core functionalities
├── hey_jimmy.onnx        # Example wake word model (or path to it)
├── main.py               # Main application entry point
├── requirements.txt      # Python package dependencies
├── run_tests.py          # Script to execute tests
└── README.md             # This file
```

## Getting Started

### Prerequisites

*   Python 3.x
*   Pip (Python package installer)
*   PortAudio (required by PyAudio)
    *   On Debian/Ubuntu: `sudo apt-get install portaudio19-dev`
    *   On macOS: `brew install portaudio`

### Installation

1.  **Clone the repository (if applicable):**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *   The `requirements.txt` includes `pvporcupine` for Picovoice wake word support.
    *   The `requirements.txt` includes `psutil` for the System Info module.
    *Note: `requirements.txt` may need to be updated or created based on all necessary packages like `openwakeword`, `pyaudio`, `whisper`, `pyttsx3`, etc.*

4.  **Wake Word Model:**
    Ensure you have a wake word model file (the configured wake word model (e.g., `./hey_jimmy.onnx` for OpenWakeWord or a `.ppn` file for Picovoice) referenced in `main.py`). You might need to download or train one if it's not included or if you wish to use a different wake word.

## First-Time Setup

When you run the application for the first time, a setup script (`first_run_setup.py`) will automatically execute to configure your wake word.

### Prerequisites for Setup:
1.  **Picovoice Access Key**: You MUST have a Picovoice Access Key for the wake word functionality to work with the pre-generated options.
    *   Obtain a free Access Key from the [Picovoice Console](https://console.picovoice.ai/).
    *   Set it as an environment variable named `PICOVOICE_ACCESS_KEY` before running the application for the first time.
      (e.g., `export PICOVOICE_ACCESS_KEY='YourActualKeyHere'` on Linux/macOS or set it in your system environment variables on Windows).

### The Setup Process:
1.  The script will greet you and confirm if your `PICOVOICE_ACCESS_KEY` is found.
2.  It will then present a list of pre-configured wake word options (e.g., "Computer", "Jarvis").
3.  You will be asked to choose your preferred wake word.
    *   **Voice Input**: The script will attempt to use your voice to select the wake word. It will prompt you to speak your choice (e.g., the name of the wake word like "Computer", or its number like "Option One").
    *   **Transcription**: Your spoken audio will be captured and transcribed using the Whisper STT engine.
    *   **Retry Logic**: If your speech isn't clearly understood or doesn't match an option, you'll be asked to try again a few times.
    *   **Typed Fallback**: If voice input fails after these retries, or if there's an issue with audio capture/transcription, the script will automatically fall back to asking you to type the number corresponding to your choice.
4.  Once selected, the script will save your choice to `user_settings.json` in the project directory.
5.  **TTS Voice Selection**: After setting up your wake word, you'll be guided to choose a Text-to-Speech (TTS) voice for the assistant.
    *   Available voices (which depend on your system's TTS engines) will be listed.
    *   You can select a voice using voice input (e.g., saying the voice name or option number) or by typing the corresponding number if voice input isn't clear.
    *   Your chosen voice ID will be saved in `user_settings.json`.
    *   The script will attempt to use this new voice for its final messages.
6.  You will then be prompted to **restart the main application** for the new wake word and TTS voice to take effect.

**Note**: If the `PICOVOICE_ACCESS_KEY` is not set, the setup script will instruct you to set it and exit. The main application will not run fully until setup is complete.

### Running the Assistant

To start the voice assistant, run:
```bash
python main.py
```
The assistant will initialize, load modules, and start listening for the wake word.

## Running Tests

To execute the test suite:
```bash
python run_tests.py
```
Alternatively, if tests are set up for `pytest`:
```bash
pytest
```

## How it Works

1.  The `main.py` script initializes the `Assistant`.
2.  The `Assistant` loads intent modules from the `modules/` directory. Each module registers phrases it can understand and the functions to call.
3.  The `VoiceCore` (`core/engine.py`) starts listening. It uses `PyAudio` to capture microphone input.
4.  The configured wake word engine (OpenWakeWord or Picovoice) continuously processes the audio stream. When the wake word (e.g., "Hey Jimmy") is detected, the `VoiceCore` triggers its `on_wake_word` callback.
5.  The assistant (via `handle_wake_word` in `main.py`) typically gives an audio cue (e.g., "Yes?") using the TTS engine.
6.  The `VoiceCore` then records the audio following the wake word until a period of silence.
7.  This recorded audio is transcribed into text using `whisper`.
8.  The transcribed command is passed to the `on_command` callback in `VoiceCore`.
9.  The `Assistant` (via `handle_command` in `main.py`) iterates through its loaded intents. If the command matches a registered intent phrase, the associated action function is executed.
10. Action functions within modules perform their tasks and can use `core.tts.speak()` to provide voice feedback.

## Modules and Capabilities

The assistant's capabilities are extended by modules found in the `modules/` directory. These modules define specific voice commands and actions. Here's an overview of the identified modules:

*   **`general.py`**:
    *   Tells the current time.
    *   Can run a system self-test (executes `run_tests.py`).
*   **`ping.py`**:
    *   Pings a specified IP address to check for connectivity.
    *   *Potential Enhancement*: Could be updated to ping named systems from `modules/configs/systems_config.json`.
*   **`wol.py` (Wake-on-LAN)**:
    *   Wakes up registered computers by sending a "magic packet."
    *   Uses `modules/configs/systems_config.json` to map system names (e.g., "PC1") to their MAC addresses.
*   **`find_devices.py`**:
    *   (Presumed) Scans the local network to discover active devices.
*   **`server.py`**:
    *   (Presumed) Functionality related to interacting with or managing a server. The exact capabilities would need to be reviewed in its source code.
*   **`speedtest.py`**:
    *   (Presumed) Performs an internet speed test.
*   **`system_info.py`** (uses `psutil` library):
    *   Provides information about the system's hardware and operating system.
    *   **Capabilities**:
        *   Get current CPU utilization percentage.
        *   Get current memory (RAM) usage (total, used, free, percentage).
        *   Get disk usage for a specified path (defaults to root `/` or `C:\`) showing total, used, free, and percentage.
        *   Get system uptime (how long the system has been running).
        *   Get current system load average (more relevant for Linux/macOS).
        *   Provide an overall system status summary.
    *   **Example Voice Commands**:
        *   "system status"
        *   "cpu usage"
        *   "memory usage"
        *   "disk space" (for default drive)
        *   "disk space for /home" (example for specific path, requires argument parsing in `main.py` to be effective)
        *   "system uptime"
        *   "system load"

### Module Configuration

Some modules, like `wol.py`, rely on a configuration file located at `modules/configs/systems_config.json`. This JSON file allows you to define friendly names for your network devices and store their MAC addresses and IP addresses.

Example `systems_config.json`:
```json
{
    "PC1": {
        "mac_address": "00:1A:2B:3C:4D:5E",
        "ip_address": "192.168.1.100"
    },
    "ServerAlpha": {
        "mac_address": "11:22:33:44:55:66",
        "ip_address": "192.168.1.101"
    }
}
```
You can edit this file to add or modify the devices your voice assistant can interact with.

---

## Extending Functionality

To add new voice commands:
1.  Create a new Python file in the `modules/` directory (e.g., `my_new_module.py`).
2.  Inside this file, define functions that will perform the desired actions.
3.  Create a function named `register_intents()` that returns a dictionary. The keys should be the voice command phrases (in lowercase), and the values should be the action functions you defined.
    Example:
    ```python
    from core.tts import speak

    def my_custom_action():
        speak("Executing your custom command!")
        # ... do something ...

    def another_action():
        speak("Doing something else.")
        # ... do something else ...

    def register_intents():
        return {
            "do my custom thing": my_custom_action,
            "perform another task": another_action
        }
    ```
4.  The assistant will automatically load your new module and its intents when it starts.

Ensure any new dependencies for your module are added to `requirements.txt`.

## Wake Word Configuration (for Developers)

The voice assistant uses Picovoice Porcupine for wake word detection when configured via the first-run setup. This allows for high-accuracy, low-resource custom wake words.

### Adding New Pre-Generated Wake Words:
1.  **Picovoice Console**: Go to the [Picovoice Console](https://console.picovoice.ai/).
2.  **Create Wake Word(s)**: Use the Porcupine wake word tool to type your desired wake word(s) (e.g., "Hey Assistant", "Hello Friend").
3.  **Download Model Files**: Download the `.ppn` model file for each wake word. Make sure to select the correct platform (e.g., Raspberry Pi, Linux, Windows, macOS) that matches where you'll run this application.
4.  **Place Model Files**: Put the downloaded `.ppn` files into the `wakeword_models/` directory in the project root. (Create this directory if it doesn't exist).
5.  **Update `first_run_setup.py`**: Open `first_run_setup.py` and modify the `AVAILABLE_WAKE_WORDS` list. Add a new dictionary for each wake word you've added, specifying its display `name` and the relative `model_file` path (e.g., `{\"name\": \"My New Word\", \"model_file\": \"wakeword_models/My_New_Word.ppn\"}`).

Users will then see these new options during the first-time setup.

### User Configuration File (`user_settings.json`)
User preferences, including the chosen wake word engine and model path, are stored in `user_settings.json`. This file is managed by `core/user_config.py` and is now located in your Windows user profile at `%APPDATA%/voice_assistant/user_settings.json` (not the project root).

**Configuration keys:**
- `first_run_complete`: Whether the initial setup has been completed.
- `chosen_wake_word_engine`: e.g., "openwakeword" or "picovoice".
- `chosen_wake_word_model_path`: Path to the wake word model file (e.g., .onnx or .ppn).
- `picovoice_access_key_is_set_env`: Confirms if the environment variable was detected during setup.
- `chosen_tts_voice_id`: The ID of the TTS voice selected by the user during setup.
- `api_keys`: Dictionary for storing API keys for integrations (e.g., weather).
- `language`: User language preference (e.g., "en").

**How the config works:**
- The config file is created automatically on first run if it does not exist.
- All changes to user preferences are saved through the application; manual editing is possible but not recommended.
- Before overwriting, a backup is created at `%APPDATA%/voice_assistant/user_settings_backup.json`.
- If the config is missing keys, defaults are filled in automatically.

**Advanced:**
- You can safely delete or reset the config file to restore defaults; the app will recreate it.
- Sensitive data (like API keys) is never logged.

### Future Plans for Custom Wake Words (Developer Note)
The current system uses pre-generated wake word models chosen by the user during setup. Future development aims to explore more dynamic custom wake word creation, potentially allowing users to:
1.  Speak a brand new, never-before-heard phrase, and have the system attempt to create a wake word model for it on the fly (highly complex, research-level task).
2.  Type their desired wake word during setup, with the system then attempting to configure itself to use it, possibly by guiding the user through a service like Picovoice Console for model generation and placement (complex integration task).
These are significant undertakings and are noted here for future consideration.
