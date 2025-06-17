import threading
import asyncio
import re
import platform
import datetime # Added for global datetime access
from typing import Optional

from wakeword_detector import run_wakeword_async

from modules.audio_utils import record_audio_async
from modules.stt_service import initialize_stt, transcribe_audio_async
from modules.tts_service import initialize_tts, text_to_speech_async
from modules.weather_service import initialize_weather_service, get_weather_async
from modules.llm_service import initialize_llm, get_llm_response
from modules.intent_classifier import initialize_intent_classifier, detect_intent_async
from modules.reminder_utils import parse_reminder, parse_list_reminder_request
from modules.db_manager import (
    initialize_db,
    save_reminder_async,
    get_reminders_for_date_async,
    reminder_check_loop,
)
from modules.gui_utils import show_reminders_gui  # type: ignore
from modules.retrain_utils import trigger_model_retraining_async, parse_retrain_request
from modules.contractions import normalize_text
from modules.error_handling import async_error_handler
from typing import Callable, Dict, Awaitable
from modules.greeting_module import get_greeting, get_goodbye
from modules.calendar_utils import add_event_to_calendar # get_calendar_file_path not used directly here yet

# --- Modularized interaction logic ---

# --- Intent Handling Registry ---
INTENT_HANDLERS: Dict[str, Callable[[str], Awaitable[str]]] = {}


def intent_handler(intent_name: str):
    def decorator(func: Callable[[str], Awaitable[str]]):
        INTENT_HANDLERS[intent_name] = func
        return func

    return decorator


@intent_handler("cancel_task")
async def handle_cancel_task(normalized_transcription: str) -> str:
    response = "Okay, cancelling that. (Note: Advanced cancel not yet implemented.)"
    await text_to_speech_async(response)
    return response


@intent_handler("calendar_query")
async def handle_calendar_query(normalized_transcription: str) -> str:
    response = "I'm not yet connected to your calendar, but I can set reminders."
    await text_to_speech_async(response)
    return response


@intent_handler("greeting")
async def handle_greeting_intent(normalized_transcription: str) -> str:
    response = get_greeting()
    print(f"Assistant (greeting): {response}")
    await text_to_speech_async(response)
    return response


@intent_handler("goodbye")
async def handle_goodbye_intent(normalized_transcription: str) -> str:
    response = get_goodbye()
    print(f"Assistant (goodbye): {response}")
    await text_to_speech_async(response)
    print("Shutting down assistant as requested by user.")
    await text_to_speech_async("Shutting down assistant as requested by user.")
    import sys
    sys.exit(0)


@async_error_handler()
async def process_command(transcription: str):
    normalized_transcription = normalize_text(transcription)
    print(f"Processing command: {normalized_transcription}")
    # Removed verbose "Processing command" TTS
    intent = await detect_intent_async(normalized_transcription)

    # Special handling for retrain_model as it's combined with parse_retrain_request
    if intent == "retrain_model" or parse_retrain_request(normalized_transcription):
        response = "Starting model retraining. This may take a few minutes."
        await text_to_speech_async(response)
        try:
            _success, retrain_msg = await trigger_model_retraining_async()
        except Exception as e:
            retrain_msg = f"Retraining failed due to an error: {e}"
        print(retrain_msg)
        await text_to_speech_async(retrain_msg)
        return  # Exit early as speech is handled

    handler = INTENT_HANDLERS.get(intent)  # Check registered handlers first
    response_text = "" # Initialize for clarity, though handlers return their own

    if handler:
        # Handlers are responsible for their own TTS and returning the spoken text
        response_text = await handler(normalized_transcription)
    else:  # Fallback to LLM
        print("Sending to LLM for general query or unhandled/low-confidence intent...")
        llm_response = await get_llm_response(input_text=normalized_transcription)
        if (
            not llm_response
            or "don't understand" in llm_response.lower()
            or "sorry" in llm_response.lower()
        ):  # Broader check for LLM uncertainty
            response_text = (
                "I'm sorry, I didn't understand that. "
                "You can ask me to set reminders, check the weather, or answer questions. "
                "Try rephrasing your request or say 'help' for examples."
            )
        else:
            response_text = llm_response
        
        print(f"Assistant (LLM): {response_text}")
        if response_text: # Ensure there's something to say from LLM path
            await text_to_speech_async(response_text)
    # Individual handlers (including LLM path) now manage their own TTS.
    # The 'response_text' variable here primarily holds what was spoken for logging/debugging if needed.


# Define placeholder handlers for intents previously in the first process_command
@intent_handler("set_reminder")
async def handle_set_reminder_intent(normalized_transcription: str) -> str:
    reminder = parse_reminder(normalized_transcription)
    if reminder:
        await save_reminder_async(reminder["task"], reminder["time"])
        # Add reminder to calendar
        add_event_to_calendar(reminder["task"], reminder["time"])
        response = f"Okay, I've set a reminder for '{reminder['task']}' at {reminder['time'].strftime('%I:%M %p on %A, %B %d')} and added it to your calendar."
    else:
        response = "I couldn't quite understand the reminder. Please try saying something like 'remind me to call John tomorrow at 2 pm'."
    await text_to_speech_async(response)
    return response


@intent_handler("list_reminders")
async def handle_list_reminders_intent(normalized_transcription: str) -> str:
    target_date = parse_list_reminder_request(normalized_transcription or "")
    if target_date:
        reminders_found = await get_reminders_for_date_async(target_date)
        date_str = target_date.strftime("%A, %B %d, %Y")
        if reminders_found:
            response = f"Here are your reminders for {date_str}: "
            for r in reminders_found:
                response += f"{r['task']} at {r['time'].strftime('%I:%M %p')}. "
        else:
            response = f"You have no reminders scheduled for {date_str}."
        threading.Thread(
            target=show_reminders_gui, args=(reminders_found, date_str), daemon=True
        ).start()
    else:
        response = "I couldn't understand which date you want reminders for. Please specify a day like 'today', 'tomorrow', or a specific date."
    await text_to_speech_async(response)
    return response


@intent_handler("get_weather")
async def handle_get_weather_intent(normalized_transcription: str) -> str:
    location_name: Optional[str] = None
    use_current_location = False
    response = ""
    my_area_phrases = [
        "my area",
        "here",
        "current location",
        "around me",
        "local weather",
    ]
    simple_weather_queries = [
        "what's the weather",
        "weather today",
        "weather now",
        "tell me the weather",
        "weather",
    ]
    location_match = re.search(
        r"(?:weather in|weather for|weather at|weather like in)\s+([A-Za-z\s]+)",
        normalized_transcription.lower(),
    )
    extracted_location = location_match.group(1).strip() if location_match else None
    if extracted_location:
        if extracted_location.lower() in [p.lower() for p in my_area_phrases]:
            use_current_location = True
        else:
            location_name = extracted_location
    else:
        transcription_lower_stripped = normalized_transcription.lower().strip()
        is_simple_query = transcription_lower_stripped in simple_weather_queries
        is_my_area_query = any(
            phrase in transcription_lower_stripped for phrase in my_area_phrases
        )
        if is_my_area_query or is_simple_query:
            use_current_location = True
        else:
            response = "Which location's weather are you interested in? For example, say 'what is the weather in London' or 'what is the weather in my area'."
            await text_to_speech_async(response)
            return response
    weather_data = None
    if use_current_location:
        print("Fetching weather for current location...")
        await text_to_speech_async("Fetching weather for current location...")
        weather_data = await get_weather_async(None)
        if weather_data:
            response = f"The current weather in {weather_data['city']} is {weather_data['description']} with a temperature of {weather_data['temp']:.1f} degrees Celsius."
            # Add weather as calendar event (add_event_to_calendar is already globally imported)
            today = datetime.datetime.now().replace(hour=9, minute=0, second=0, microsecond=0) # Use globally imported datetime
            add_event_to_calendar(f"Weather in {weather_data['city']}: {weather_data['description']}", today, description=f"Temperature: {weather_data['temp']:.1f}°C")
            response += " I've also added this to your calendar."
        else:
            response = "Sorry, I couldn't determine your current location or fetch the weather for it. Please check your internet connection or try specifying a city."
    elif location_name:
        print(f"Fetching weather for {location_name}...")
        await text_to_speech_async(f"Fetching weather for {location_name}...")
        weather_data = await get_weather_async(location_name)
        if weather_data:
            response = f"The current weather in {weather_data['city']} is {weather_data['description']} with a temperature of {weather_data['temp']:.1f} degrees Celsius."
        else:
            response = f"Sorry, I couldn't fetch the weather for {location_name}. Please ensure the API key is set up and the location is valid."
    if not response:
        response = "I'm not sure which location you're asking about for the weather. Please specify, like 'weather in London' or 'weather in my area'."
    await text_to_speech_async(response)
    return response


@intent_handler("add_calendar_event")
async def handle_add_calendar_event_intent(normalized_transcription: str) -> str:
    # Example: "add meeting with John on June 20th at 3pm"
    # Ensure dateparser and re are imported (they are at the top of the context file)
    import dateparser 
    # import re # Already imported at the top

    response = ""
    # Try to extract event title and date/time using various patterns
    patterns = [
        r"add (.+?)(?:\s+on|\s+at|\s+for)\s+(.+)", # "add event on date", "add event at time", "add event for date"
        r"schedule (.+?)(?:\s+on|\s+at|\s+for)\s+(.+)", # "schedule event on date", etc.
        r"put (.+?)(?:\s+on my calendar|\s+in my calendar)(?:\s+for|\s+on|\s+at)\s+(.+)" # "put event on my calendar for date"
    ]
    
    summary = None
    date_str = None

    for pattern in patterns:
        match = re.search(pattern, normalized_transcription, re.IGNORECASE)
        if match:
            summary = match.group(1).strip()
            # Clean up summary: remove "called X" if it's part of the date string, or rephrase
            called_match = re.search(r"(.+?)\s+called\s+(.+)", summary, re.IGNORECASE)
            if called_match:
                 summary = f"{called_match.group(1).strip()}: {called_match.group(2).strip()}"
            date_str = match.group(2).strip()
            break 

    if summary and date_str:
        start = dateparser.parse(date_str, settings={'PREFER_DATES_FROM': 'future'})
        if start:
            response = add_event_to_calendar(summary, start) # add_event_to_calendar returns a string
        else:
            response = "Sorry, I couldn't understand the date and time for the event. Please try again, like 'add meeting on June 20th at 3pm'."
    else:
        response = "Please specify the event and date, for example: 'add meeting with John on June 20th at 3pm' or 'schedule project update for next Tuesday at 2pm'."
    await text_to_speech_async(response)
    return response


async def handle_interaction():
    greeting = get_greeting()
    print(f"Assistant (speaking): {greeting}")
    await text_to_speech_async(greeting)
    audio_data = await record_audio_async()
    transcription = await transcribe_audio_async(audio_data)
    if not transcription or not transcription.strip():
        print("No speech detected after greeting.")
        await text_to_speech_async("I didn't catch that. If you need something, please call me again.")
        return
    print(f"User said: {transcription}")
    await process_command(transcription)


def run_assistant():
    # --- Initialization ---
    print("Initializing services...")
    loop = asyncio.get_event_loop()
    initialize_stt()
    initialize_tts()
    initialize_weather_service()
    initialize_llm()
    initialize_intent_classifier()
    initialize_db()

    print("Services initialized. You can now interact with the assistant.")

    # Start reminder check loop as a background task
    loop.create_task(reminder_check_loop(text_to_speech_async))

    # --- Main loop ---
    async def main_loop():
        while True:
            try:
                # Wake word detection
                print("Waiting for wake word...")

                # Define a simple callback that sets an event when the wake word is detected
                wake_event = asyncio.Event()

                def on_wakeword_detected():
                    print("Wake word detected (callback)!")
                    loop = asyncio.get_event_loop()
                    if loop.is_running(): # Ensure loop is running before calling call_soon_threadsafe
                        loop.call_soon_threadsafe(wake_event.set)
                await run_wakeword_async(callback=on_wakeword_detected)
                await wake_event.wait()
                await handle_interaction() # Use the consolidated interaction logic

            except Exception as e:
                print(f"Error in main loop: {e}")

    # Run the main loop
    loop.run_until_complete(main_loop())


if __name__ == "__main__":
    # For Windows, set the policy to allow more threads if needed
    if platform.system() == "Windows":
        import nest_asyncio

        nest_asyncio.apply()

    run_assistant()
