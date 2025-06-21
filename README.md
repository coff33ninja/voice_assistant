# Voice Assistant Project

## ⚠️ Work in Progress ⚠️

This is a personal voice assistant project currently under active development. Many features are implemented, but it's still evolving.

This project is a Python-based voice assistant capable of understanding voice commands, performing tasks like setting reminders, fetching weather information, and answering general questions.

## Overview

The voice assistant uses a combination of local and cloud-based services for its functionalities:

* **Wakeword Detection**: Uses Porcupine (primary) or Precise (fallback) for detecting the "Hey Mika" wakeword.
* **Speech-to-Text (STT)**: Utilizes WhisperX for transcribing spoken audio to text.
* **Intent Classification**: Employs a fine-tuned **joint intent and slot-filling model (based on DistilBERT)** to understand the user's intent and extract key entities (like task, time, location) from the transcribed text.
* **Text-to-Speech (TTS)**: Uses Coqui TTS to generate spoken responses.
* **Language Model (LLM)**: Leverages Ollama (with a model like Llama 2) for handling general queries and providing conversational responses.
* **Task-Specific Modules**: Includes dedicated modules for weather, reminders, and potentially more in the future.
* **Calendar**: Generates and manages an `.ics` calendar file for reminders, weather entries, and direct event additions.

### Entity Extraction Details

The NLU model is trained to extract the following key entity types from user commands, enabling a more nuanced understanding of requests:

*   `subject`: The main description of a task or event (e.g., "call mom", "project meeting").
*   `time_phrase`: Natural language expressions of time, date, or duration (e.g., "tomorrow at 3pm", "in an hour", "next Monday"). This is typically parsed by the system to determine specific datetime objects.
*   `date_reference`: Specific references to dates, often relative or named (e.g., "today", "tomorrow", "July 10th").
*   `time`: Specific references to times of day (e.g., "5 PM", "morning", "noon").
*   `location`: Geographical places or named locations (e.g., "London", "the office", "current location").
*   `contact_name`: Names of people or groups involved (e.g., "John", "Marketing team", "Dr. Smith").
*   `duration`: Specific lengths of time (e.g., "1 hour", "for 30 minutes", "two days").
*   `topic`: The subject matter of a general knowledge question (e.g., "capital of France", "how photosynthesis works").
*   `item_to_add` / `item_to_remove`: Specific items for list management commands (e.g., "milk" for a shopping list).
*   `text_to_translate` / `target_language`: The text to be translated and the language to translate it into.

The model's ability to accurately extract these entities is directly dependent on comprehensive annotations within the `intent_data/intent_dataset.csv` training data.

## Features

* **Wakeword Activation**: Listens for a wakeword to start interaction.
* **Set Reminders**: "Remind me to call mom tomorrow at 2:30 pm" (also adds to calendar).
* **List Reminders**: "What are my reminders for today?" (with GUI display).
* **Get Weather Information**: "What’s the weather like today?" or "What's the weather in London?"
* **Add Calendar Events**: "Add meeting with John on June 20th at 3pm".
* **General Question Answering**: "What’s the capital of France?"
* **Model Retraining**: "Retrain the model" (triggers retraining of the intent classifier).
* **Text Normalization**: Expands contractions (e.g., "what's" to "what is") and corrects common misspellings (e.g., "gonna" to "going to").
* **Calendar File Generation**: Creates an `assistant_calendar.ics` file that can be imported/synced with most calendar applications.
* **Greeting/Goodbye**: Handles basic conversational openings and closings.
* **Chat with AI**: Allows open-ended conversations with the configured Ollama LLM, with an option to save the interaction.
* **Enhanced Command Understanding**: Actively extracts details (entities) from commands, leading to more reliable interpretation of requests for reminders, weather, etc.

## Project Structure

```
voice_assistant/
├── assets/                   # Audio assets (e.g., XTTS voice cloning)
│   └── sample_speaker.wav    # Sample audio (3–10s) for XTTS cloning
├── models/                   # Stores downloaded models, API keys, DB
├── modules/                  # Core logic for different functionalities
│   ├── __init__.py
│   ├── api_key_setup.py
│   ├── audio_utils.py
│   ├── calendar_utils.py   # Calendar file (.ics) management
│   ├── config.py             # Configuration variables
│   ├── config_env.py         # Environment-specific configurations
│   ├── contractions.py       # Text normalization
│   ├── dataset.py            # Loads intent data from intent_data/intent_dataset.csv
│   ├── db_manager.py         # Database interactions for reminders
│   ├── db_setup.py
│   ├── device_detector.py    # Network device discovery and management
│   ├── download_and_models.py # Downloads TTS/Precise models
│   │                           # Plays sample speaker audio for XTTS
│   ├── error_handling.py     # Centralized error handling utilities
│   ├── file_watcher_service.py # Monitors files for changes (e.g., config, data)
│   ├── greeting_module.py    # Handles greetings and goodbyes
│   ├── gui_utils.py          # Simple Tkinter GUI for reminders
│   ├── install_dependencies.py # Dependency installer
│   ├── intent_classifier.py  # Intent classification and entity extraction logic
│   ├── intent_logic.py       # Core intent handling logic
│   ├── joint_model.py        # Defines the joint intent and slot-filling model architecture
│   ├── llm_service.py        # LLM interaction
│   ├── model_training.py     # Script to train the intent model
│   ├── normalization_data/   # Directory for normalization and augmentation resources
│   │   ├── common_misspellings_map.json
│   │   ├── contractions_map.json
│   │   ├── custom_dictionary.txt
│   │   ├── ... (augmented/merged dictionary files)
│   ├── not-implemented/      # Temporarily removed features under development
│   │   ├── configs/
│   │   ├── device_manager.py
│   │   ├── find_devices.py
│   │   ├── general.py
│   │   ├── ping.py
│   │   ├── server.py
│   │   ├── shutdown.py
│   │   ├── speedtest.py
│   │   ├── system_info.py
│   │   ├── weather.py
│   │   ├── wol.py
│   │   ├── __init__.py
│   │   └── __pycache__/
│   ├── ollama_setup.py
│   ├── reminder_utils.py     # Parsing reminder requests
│   ├── retrain_utils.py      # Utilities for triggering retraining
│   ├── stt_model_selection.py # STT model selection and testing
│   ├── stt_service.py        # Speech-to-text
│   ├── tts_service.py        # Text-to-speech
│   │                           # Handles XTTS with assets/sample_speaker.wav
│   ├── utils.py
│   ├── weather_service.py    # Weather fetching
│   ├── whisper_setup.py
│   └── whisperx_setup.py     # WhisperX initial setup and test
├── intent_data/              # Data for intent classification and responses
│   ├── intent_dataset.csv    # CSV: utterances mapped to intents and annotated with entities (see USER_INTENTS.md for intent details). Entity annotations are crucial for training the slot-filling capabilities of the model.
│   ├── intent_dataset_augmented.csv # Augmented intent dataset with paraphrases, misspellings, etc.
│   ├── intent_responses.csv  # CSV: predefined responses mapped to intents
│   └── augmentation_stats.json # JSON: stats and logs from the latest data augmentation
├── scripts/                  # Utility scripts (e.g., data conversion, maintenance, helper tools)
│   ├── augment_dictionaries.py # Augments and merges all dictionary files
│   ├── augment_intent_dataset.py # Augments the intent dataset with paraphrasing, misspellings, and more
│   ├── intent_validator.py   # Script to validate intent data consistency and model retraining
│   ├── models/
│   │   ├── intent_dataset.csv
│   │   └── intent_responses.csv
│   ├── __init__.py
│   ├── validate_and_clean_sentences.py # Validates and auto-corrects sentences in the intent dataset (auto-run before augmentation)
│   └── archive_augmented_data.py       # Archives previous augmented data as checkpoints before new augmentation
├── setup_assistant.py        # Main setup script
├── voice_assistant.py        # Main application script
├── USER_INTENTS.md           # Describes available user commands/intents
├── wakeword_detector.py      # Wakeword detection logic
└── README.md                 # This file
```

## Setup

1. **Prerequisites**:

   * Python (3.7-3.11 recommended).
   * `pip` (Python package installer).
   * System dependencies like `ffmpeg`, `libsndfile`, `portaudio` (see `modules/install_dependencies.py` for platform-specific instructions).
   * **Windows Specific**: Microsoft Visual C++ Redistributable (Visual Studio 2015-2022). This is required for some Python packages like ONNXRuntime. You can download it from the official Microsoft site:
     * For 64-bit systems: https://aka.ms/vs/17/release/vc_redist.x64.exe
     * For 32-bit systems: https://aka.ms/vs/17/release/vc_redist.x86.exe
   * Ollama installed and running (e.g., `ollama serve`).

2. **Clone the repository (if applicable) or ensure you are in the project's root directory.**

3. **Run the setup script**:

   ```bash
   python setup_assistant.py
   ```

   This script will:

   * Install Python dependencies.

   * Install system dependencies (attempt to).

   * Download necessary models (TTS, wakeword).

   * Guide through TTS model selection (will attempt to list available Coqui TTS models and download the chosen one).

   * Allow configuration of TTS speaking speed.

   * Prompt for API keys (Picovoice for Porcupine, OpenWeather).

   * Set up the WhisperX STT engine and test it.

   * Initialize the database.

   * Create a sample dataset for intent classification.

   * Fine-tune the intent classification model.

   > **Note on XTTS Models**
   > If you select an XTTS model (e.g., `tts_models/multilingual/multi-dataset/xtts_v2`), ensure a voice sample file named `sample_speaker.wav` exists in the `assets/` directory.
   > This file is used for:
   >
   > * Voice cloning during setup-time test playback
   > * Runtime responses with XTTS-based TTS
   >
   > Recommended: A clean, clear 3–10 second WAV file of a single speaker talking.

4. **PyTorch Lightning Checkpoint Upgrade**:
   The setup script attempts to automatically apply a PyTorch Lightning checkpoint upgrade for the WhisperX model. This helps prevent a recurring message about the upgrade. If the automatic attempt fails or is not applicable, you might still see the message and can try running the suggested command manually if needed.

## Running the Assistant

Once the setup is complete, you can run the voice assistant using:

```bash
python voice_assistant.py
```

Say the wakeword (e.g., "Hey Mika," depending on your trained model) to interact with the assistant.

## Available Commands

For a list of commands the assistant currently understands and examples of how to phrase them, please see the USER\_INTENTS.md file.

The primary intents include setting reminders, listing reminders, getting weather information, general question answering, and triggering model retraining.

## Future Work & Not Implemented

Please note: A folder named `not-implemented` containing modules for features like calendar integration, music control, etc., was part of earlier versions. This folder has been temporarily removed for a focused refactor and will be re-integrated with improved functionality at a later date.

## Key Modules

* `voice_assistant.py`: Main application orchestrator.
* `wakeword_detector.py`: Handles wakeword detection using Precise or Porcupine.
* `modules/stt_service.py`: Transcribes audio to text using WhisperX.
* `modules/intent_classifier.py`: Determines user intent and extracts relevant entities from the command.
* `modules/llm_service.py`: Handles general queries via Ollama.
* `modules/tts_service.py`: Converts text responses to speech. Handles XTTS models by using assets/sample\_speaker.wav if configured.
* `modules/contractions.py`: Normalizes user input text.
* `modules/reminder_utils.py` & `modules/db_manager.py`: Manage reminders.
* `modules/weather_service.py`: Fetches weather data.
* `setup_assistant.py`: Comprehensive setup script for all components.
* `modules/download_and_models.py`: Handles downloading of TTS and Precise models, including sample playback for TTS which uses assets/sample\_speaker.wav for XTTS models.

## 🛠️ Data & Resource Augmentation Pipeline

This project uses a robust, automated augmentation pipeline for both training data and dictionary resources. This ensures that all models and logic always use the freshest, most comprehensive data available.

### Automated Augmentation Steps

**Before every model training or retraining:**
- The following scripts are run automatically:
  - `scripts/augment_dictionaries.py`: Updates/augments all dictionary files (contractions, misspellings, synonyms, normalization, etc.). Both original and augmented versions are saved and available for use.
  - `scripts/augment_intent_dataset.py`: Augments the intent dataset with paraphrases, contractions, misspellings, and more. The augmented dataset is always used for training if available.

**Sentence Validation & Cleaning:**
- Before any augmentation or training, `scripts/validate_and_clean_sentences.py` is run to auto-correct and clean all sentences in the intent dataset. Only the cleaned dataset is used for augmentation and training, ensuring high data quality.

**Checkpointing:**
- Before each augmentation run, `scripts/archive_augmented_data.py` archives the previous batch of augmented data as a checkpoint for traceability and rollback.

**Intent Dataset Handling:**
- Training and retraining always use the latest cleaned and augmented dataset (`intent_dataset_cleaned.csv` → `intent_dataset_augmented.csv`) if present. If not, the cleaning and augmentation steps are triggered automatically.

**Dictionary Handling:**
- All modules that use dictionaries (e.g., `modules/contractions.py`, normalization, etc.) are designed to load the augmented version if it exists, falling back to the original if not.
- A merged view (`merged_dictionaries.json`) is also generated for downstream use.

### Stats & Logging
- Both augmentation scripts log stats (counts, file paths, etc.) for transparency and debugging.
- Hardware info (CPU/GPU) is logged and included in augmentation stats.

### Key Scripts

- `scripts/augment_dictionaries.py`: Augments and merges all dictionary resources. Always run before training.
- `scripts/augment_intent_dataset.py`: Augments the intent dataset with paraphrasing, misspellings, and more.
- `scripts/intent_validator.py`: Validates intent data consistency and can be run after augmentation for sanity checks.

### Design Strategy

- **Always Up-to-Date:** All training and inference steps use the most recent, augmented data and resources.
- **Non-Destructive:** Original files are never overwritten; augmented versions are saved separately.
- **Extensible:** New dictionary/resource types can be added to the augmentation pipeline with minimal changes.
- **Centralized Automation:** All augmentation and validation steps are triggered automatically from the training pipeline (`modules/model_training.py`), so no manual intervention is needed.

**Troubleshooting & Environment Notes**

- **Java Requirement for LanguageTool:**
  - The sentence validation/cleaning step requires Java 17 or higher. If you see errors about Java version (e.g., `SystemError: Detected java 1.8. LanguageTool requires Java >= 17`), download and install the latest Java JDK from [Adoptium](https://adoptium.net/) or [Oracle](https://www.oracle.com/java/technologies/downloads/). After installation, restart your terminal/IDE and ensure `java -version` reports 17 or higher. {[Java 17 x64] (https://download.oracle.com/java/17/archive/jdk-17.0.12_windows-x64_bin.exe)

- **Module Import Error in Augmentation or Setup:**
  - If you see `ModuleNotFoundError: No module named 'modules'` during setup or when running `voice_assistant.py`, it means Python cannot find your project modules. This can happen if:
    - You run scripts from a subdirectory instead of the project root.
    - The `PYTHONPATH` is not set to include your project root.
  - **How to fix:**
    1. Always run `python setup_assistant.py` and `python voice_assistant.py` from the project root directory (where `modules/` is located).
    2. If you must run from another directory, set the `PYTHONPATH` to your project root as shown above.
  - The setup and main scripts are designed to work from the project root, so this error should not occur if you follow this convention.
