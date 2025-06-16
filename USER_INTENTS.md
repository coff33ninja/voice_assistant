# Voice Assistant Commands and Intents

This document outlines the types of commands the voice assistant can currently understand and process.

## Available Intents

The assistant categorizes your spoken commands into one of the following intents:

*   **Set Reminder (`set_reminder`)**
    *   Purpose: To create a new reminder for a specific task at a given time.
    *   Example Phrases:
        *   "Remind me to call mom tomorrow at 2:30 pm"
        *   "Set a reminder to buy groceries in 2 hours"
        *   "Remind me to check the oven at 7 pm"

*   **List Reminders (`list_reminders`)**
    *   Purpose: To retrieve and list reminders for a specific day.
    *   Example Phrases:
        *   "What are my reminders for today?"
        *   "Show me reminders for tomorrow"
        *   "List reminders for next Monday"

*   **Get Weather Information (`get_weather`)**
    *   Purpose: To fetch and report the current weather conditions for a specified location or the user's current location.
    *   Example Phrases:
        *   "What’s the weather like today?"
        *   "What's the weather in London?"
        *   "Tell me the weather for my area"

*   **General Question Answering (`general_query`)**
    *   Purpose: To answer general knowledge questions or handle conversational input not covered by other specific intents. This is often the fallback intent.
    *   Example Phrases:
        *   "What’s the capital of France?"
        *   "Tell me a fun fact"
        *   "How are you?"

*   **Retrain Model (`retrain_model`)**
    *   Purpose: To trigger the retraining process for the assistant's intent classification model.
    *   Example Phrases:
        *   "Retrain the model"
        *   "Start model retraining"

*   **Calendar Query (`calendar_query`)**
    *   Purpose: (Currently a placeholder) Intended for interacting with a calendar. The assistant will state it's not yet connected.
    *   Example Phrases: "What's on my calendar today?"

*   **Cancel Task (`cancel_task`)**
    *   Purpose: (Currently a placeholder) Intended for cancelling an ongoing or previous action. The assistant will acknowledge the cancellation.
    *   Example Phrases: "Cancel that", "Stop"

---
*Note: The assistant's ability to understand variations of these phrases depends on its training data and the accuracy of the Speech-to-Text (STT) engine.*