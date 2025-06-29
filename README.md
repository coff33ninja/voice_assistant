# Voice Assistant Project

This project is a Python-based voice assistant capable of understanding voice commands, performing tasks like setting reminders, fetching weather information, and answering general questions.

## Overview

The voice assistant uses a combination of local and cloud-based services for its functionalities:
*   **Wakeword Detection**: Uses Porcupine (primary) or Precise (fallback) for detecting the "Hey Mika" wakeword.
*   **Speech-to-Text (STT)**: Utilizes WhisperX for transcribing spoken audio to text.
*   **Intent Classification**: Employs a fine-tuned DistilBERT model to understand the user's intent from the transcribed text.
*   **Text-to-Speech (TTS)**: Uses Coqui TTS to generate spoken responses.
*   **Language Model (LLM)**: Leverages Ollama (with a model like Llama 2) for handling general queries and providing conversational responses.
*   **Task-Specific Modules**: Includes dedicated modules for weather, reminders, and potentially more in the future.

## Features

*   **Wakeword Activation**: Listens for a wakeword to start interaction.
*   **Set Reminders**: "Remind me to call mom tomorrow at 2:30 pm"
*   **List Reminders**: "What are my reminders for today?"
*   **Get Weather Information**: "What’s the weather like today?" or "What's the weather in London?"
*   **General Question Answering**: "What’s the capital of France?"
*   **Model Retraining**: "Retrain the model" (triggers retraining of the intent classifier).
*   **Text Normalization**: Expands contractions (e.g., "what's" to "what is") and corrects common misspellings (e.g., "gonna" to "going to").

## Project Structure

```
voice_assistant/
├── models/                   # Stores downloaded models, API keys, DB
├── modules/                  # Core logic for different functionalities
│   ├── __init__.py
│   ├── api_key_setup.py
│   ├── audio_utils.py
│   ├── config.py             # Configuration variables
│   ├── contractions.py       # Text normalization
│   ├── dataset.py            # Intent classification dataset
│   ├── db_manager.py         # Database interactions for reminders
│   ├── db_setup.py
│   ├── download_and_models.py # Downloads TTS/Precise models
│   ├── gui_utils.py          # Simple Tkinter GUI for reminders
│   ├── install_dependencies.py # Dependency installer
│   ├── intent_classifier.py  # Intent detection logic
│   ├── llm_service.py        # LLM interaction
│   ├── model_training.py     # Script to train the intent model
│   ├── reminder_utils.py     # Parsing reminder requests
│   ├── retrain_utils.py      # Utilities for triggering retraining
│   ├── stt_service.py        # Speech-to-text
│   ├── tts_service.py        # Text-to-speech
│   ├── utils.py
│   ├── weather_service.py    # Weather fetching
│   └── whisperx_setup.py     # WhisperX initial setup and test
├── setup_assistant.py        # Main setup script
├── voice_assistant.py        # Main application script
├── USER_INTENTS.md           # Describes available user commands/intents
├── wakeword_detector.py      # Wakeword detection logic
└── README.md                 # This file
```

## Setup

1.  **Prerequisites**:
    *   Python (3.7-3.11 recommended).
    *   `pip` (Python package installer).
    *   System dependencies like `ffmpeg`, `libsndfile`, `portaudio` (see `modules/install_dependencies.py` for platform-specific instructions).
    *   Ollama installed and running (e.g., `ollama serve`).

2.  **Clone the repository (if applicable) or ensure you are in the project's root directory.**

3.  **Run the setup script**:
    ```bash
    python setup_assistant.py
    ```
    This script will:
    *   Install Python dependencies.
    *   Install system dependencies (attempt to).
    *   Download necessary models (TTS, wakeword).
    *   Prompt for API keys (Picovoice for Porcupine, OpenWeather).
    *   Set up the WhisperX STT engine and test it.
    *   Initialize the database.
    *   Create a sample dataset for intent classification.
    *   Fine-tune the intent classification model.

## Running the Assistant

Once the setup is complete, you can run the voice assistant using:
```bash
python voice_assistant.py
```
Say the wakeword (e.g., "Hey Mika," depending on your trained model) to interact with the assistant.

+## Available Commands
+
+For a list of commands the assistant currently understands and examples of how to phrase them, please see the USER_INTENTS.md file.
+
+The primary intents include setting reminders, listing reminders, getting weather information, general question answering, and triggering model retraining.
+
## Future Work & Not Implemented

Please note: A folder named `not-implemented` containing modules for features like calendar integration, music control, etc., was part of earlier versions. This folder has been temporarily removed for a focused refactor and will be re-integrated with improved functionality at a later date.

## Key Modules

*   `voice_assistant.py`: Main application orchestrator.
*   `wakeword_detector.py`: Handles wakeword detection using Precise or Porcupine.
*   `modules/stt_service.py`: Transcribes audio to text using WhisperX.
*   `modules/intent_classifier.py`: Determines user intent.
*   `modules/llm_service.py`: Handles general queries via Ollama.
*   `modules/tts_service.py`: Converts text responses to speech.
*   `modules/contractions.py`: Normalizes user input text.
*   `modules/reminder_utils.py` & `modules/db_manager.py`: Manage reminders.
*   `modules/weather_service.py`: Fetches weather data.
*   `setup_assistant.py`: Comprehensive setup script for all components.