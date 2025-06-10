# Voice Assistant Project

This project is a Python-based voice assistant that listens for a wake word, processes voice commands, and performs actions based on recognized intents.

## Features

*   **Wake Word Detection**: Uses `openwakeword` to listen for a specific wake word (e.g., "Hey Jimmy").
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
    *Note: `requirements.txt` may need to be updated or created based on all necessary packages like `openwakeword`, `pyaudio`, `whisper`, `pyttsx3`, etc.*

4.  **Wake Word Model:**
    Ensure you have a wake word model file (e.g., `hey_jimmy.onnx` referenced in `main.py`). You might need to download or train one if it's not included or if you wish to use a different wake word.

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
4.  `openWakeWord` continuously processes the audio stream. When the wake word (e.g., "Hey Jimmy") is detected, the `VoiceCore` triggers its `on_wake_word` callback.
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
