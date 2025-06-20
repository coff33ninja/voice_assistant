# Voice Assistant Commands and Intents

This document outlines the types of commands the voice assistant can currently understand and process.
The assistant uses an AI model that not only identifies the general intent of your command but also tries to extract important details (entities) such as times, dates, locations, and task descriptions. This helps it understand your requests more accurately.

## Available Intents

The assistant categorizes your spoken commands into one of the following intents:

*   **Greeting (`greeting`)**
    *   Purpose: To initiate a conversation or respond to a greeting.
    *   Example Phrases:
        *   "Hello"
        *   "Hi Mika"
        *   "Good morning"

*   **Goodbye (`goodbye`)**
    *   Purpose: To end the conversation and optionally shut down the assistant.
    *   Example Phrases:
        *   "Goodbye"
        *   "See you later"
        *   "Shut down"

*   **Add Calendar Event (`add_calendar_event`)**
    *   Purpose: To add a new event directly to your calendar file.
    *   Example Phrases:
        *   "Add meeting with John on June 20th at 3pm"
        *   "Schedule project update for next Tuesday at 2pm"
        *   "Put dentist appointment on my calendar for July 1st at 10 am"

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
*Note: The assistant's ability to understand variations of these phrases and accurately extract details (entities) depends heavily on its training data (including comprehensive entity annotations in `intent_data/intent_dataset.csv`) and the accuracy of the Speech-to-Text (STT) engine.*

*   **Chat with AI (`start_chat_with_llm`)**
    *   Purpose: To engage in an open-ended conversation with the AI language model (Ollama) and optionally save the interaction.
    *   Example Phrases:
        *   "Let's chat with the AI"
        *   "I want to talk to the language model"
        *   "Start a conversation with the AI"
        *   "Open chat with AI"
        *   "Begin chat session"
    *   Interaction Flow:
        1.  The assistant confirms the start of the chat session.
        2.  The user can speak freely, and the assistant will respond using the LLM.
        3.  To end the session and save the conversation, the user says: **"stop chat and save"**.
    *   Conversation Saving:
        *   If the "stop chat and save" command is given, the entire dialogue (user inputs and AI responses) is saved.
        *   Files are stored as timestamped JSON in the `conversations/` directory at the project root.
        *   This feature is useful for reviewing interactions or gathering data for training other AI models.

---
*Note: The assistant's ability to understand variations of these phrases and accurately extract details (entities) depends heavily on its training data (including comprehensive entity annotations in `intent_data/intent_dataset.csv`) and the accuracy of the Speech-to-Text (STT) engine.*
