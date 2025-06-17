import pandas as pd

def create_dataset(dataset_path):
    data = {
        "text": [
            "Remind me to call mom tomorrow at 2:30 pm",
            "Set a reminder for meeting at 5 pm",
            "Remind me to buy groceries in 2 hours",
            "What’s on my calendar this week?",
            "Check my calendar for tomorrow",
            "Show me my schedule for Friday",
            "What’s the weather like today?",
            "Tell me the weather forecast for tomorrow",
            "Is it going to rain this afternoon in London?",
            "What are my reminders for today?",
            "Show me reminders for tomorrow",
            "List my reminders for next Monday",
            "Any reminders for July 4th?",
            "What’s the capital of France?",
            "Who won the World Cup in 2022?",
            "How does photosynthesis work?",
            "Retrain the model",
            "Update my assistant",
            "Retrain intent classifier",
            "Retrain assistant",
            "Train model",
            "Update model",
            "Cancel that",
            "Never mind",
            "Stop",
            "I don't need that anymore", # cancel_task
            "Add dentist appointment on July 5th at 10 AM",
            "Schedule a meeting for next Tuesday at 3pm called Project Update",
            "Put lunch with Sarah on the calendar for tomorrow noon", # add_calendar_event
            "Hello assistant",
            "Hi there",
            "Good morning", # greeting
            "Goodbye assistant",
            "See you later",
            "Shut down", # goodbye
        ],
        "label": [
            "set_reminder",
            "set_reminder",
            "set_reminder", # Remind me to buy groceries in 2 hours
            "calendar_query",
            "calendar_query",
            "calendar_query",
            "get_weather",
            "get_weather",
            "get_weather",
            "list_reminders",
            "list_reminders",
            "list_reminders",
            "list_reminders",
            "general_query",
            "general_query",
            "general_query",
            "retrain_model",
            "retrain_model",
            "retrain_model",
            "retrain_model",
            "retrain_model",
            "retrain_model",
            "cancel_task",   # Cancel that
            "cancel_task",
            "cancel_task",
            "cancel_task",   # I don't need that anymore
            "add_calendar_event",
            "add_calendar_event",
            "add_calendar_event",
            "greeting",
            "greeting",
            "greeting",
            "goodbye",
            "goodbye",
            "goodbye",],
    }
    df = pd.DataFrame(data)
    df.to_csv(dataset_path, index=False)
    print(f"Dataset created at {dataset_path}")
