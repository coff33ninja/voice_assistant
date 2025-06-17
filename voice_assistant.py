import threading
import asyncio
import re
import platform
import datetime # Added for global datetime access
from typing import Optional
import pandas as pd
import os

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
from modules.calendar_utils import add_event_to_calendar # get_calendar_file_path not used directly here yet

# --- Modularized interaction logic ---

# --- Intent Handling Registry ---
INTENT_HANDLERS: Dict[str, Callable[[str], Awaitable[str]]] = {}


def intent_handler(intent_name: str):
    def decorator(func: Callable[[str], Awaitable[str]]):
        INTENT_HANDLERS[intent_name] = func
        return func

    return decorator

# Load responses from CSV
RESPONSES_PATH = os.path.join(os.path.dirname(__file__), 'models/intent_responses.csv')
_responses_df = pd.read_csv(RESPONSES_PATH)
RESPONSE_MAP = dict(zip(_responses_df['intent'], _responses_df['response']))

def get_response(intent_key, **kwargs):
    resp = RESPONSE_MAP.get(intent_key, "")
    if resp and kwargs:
        try:
            return resp.format(**kwargs)
        except Exception:
            return resp
    return resp


@intent_handler("cancel_task")
async def handle_cancel_task(normalized_transcription: str) -> str:
    response = get_response("cancel_task")
    await text_to_speech_async(response)
    return response


@intent_handler("calendar_query")
async def handle_calendar_query(normalized_transcription: str) -> str:
    response = get_response("calendar_query")
    await text_to_speech_async(response)
    return response


@intent_handler("greeting")
async def handle_greeting_intent(normalized_transcription: str) -> str:
    response = get_response("greeting")
    print(f"Assistant (greeting): {response}")
    await text_to_speech_async(response)
    return response


@intent_handler("goodbye")
async def handle_goodbye_intent(normalized_transcription: str) -> str:
    response = get_response("goodbye")
    print(f"Assistant (goodbye): {response}")
    await text_to_speech_async(response)
    print("Shutting down assistant as requested by user.")
    await text_to_speech_async("Shutting down assistant as requested by user.")
    import sys
    sys.exit(0)


@intent_handler("retrain_model")
async def handle_retrain_model_intent(normalized_transcription: str) -> str:
    response = get_response("retrain_model")
    await text_to_speech_async(response)
    try:
        _success, retrain_msg = await trigger_model_retraining_async()
    except Exception as e:
        retrain_msg = get_response("retrain_model_error", error=str(e))
    print(retrain_msg)
    await text_to_speech_async(retrain_msg)
    return response


@async_error_handler()
async def process_command(transcription: str):
    normalized_transcription = normalize_text(transcription)
    print(f"Processing command: {normalized_transcription}")
    intent = await detect_intent_async(normalized_transcription)

    # Special handling for retrain_model as it's combined with parse_retrain_request
    if intent == "retrain_model" or parse_retrain_request(normalized_transcription):
        response = get_response("retrain_model")
        await text_to_speech_async(response)
        try:
            _success, retrain_msg = await trigger_model_retraining_async()
        except Exception as e:
            retrain_msg = get_response("retrain_model_error", error=str(e))
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
        if llm_response is None:
            response_text = get_response("llm_service_error")
        elif (
            not llm_response
            or "don't understand" in llm_response.lower()
            or "sorry" in llm_response.lower()
        ):
            response_text = get_response("llm_fallback_sorry")
        else:
            response_text = llm_response
        print(f"Assistant (LLM): {response_text}")
        if response_text:
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
        response = get_response("set_reminder_success", task=reminder["task"], time=reminder["time"].strftime('%I:%M %p on %A, %B %d'))
    else:
        response = get_response("set_reminder_error")
    await text_to_speech_async(response)
    return response


@intent_handler("list_reminders")
async def handle_list_reminders_intent(normalized_transcription: str) -> str:
    target_date = parse_list_reminder_request(normalized_transcription or "")
    if target_date:
        reminders_found = await get_reminders_for_date_async(target_date)
        date_str = target_date.strftime("%A, %B %d, %Y")
        if reminders_found:
            reminders_text = " ".join(f"{r['task']} at {r['time'].strftime('%I:%M %p')}." for r in reminders_found)
            response = get_response("list_reminders", date=date_str, reminders=reminders_text)
        else:
            response = get_response("list_reminders_none", date=date_str)
        threading.Thread(
            target=show_reminders_gui, args=(reminders_found, date_str), daemon=True
        ).start()
    else:
        response = get_response("list_reminders_error")
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
            response = get_response("get_weather_location_prompt")
            await text_to_speech_async(response)
            return response
    weather_data = None
    if use_current_location:
        print("Fetching weather for current location...")
        await text_to_speech_async("Fetching weather for current location...")
        weather_data = await get_weather_async(None)
        if weather_data:
            response = get_response(
                "get_weather_current",
                city=weather_data['city'],
                description=weather_data['description'],
                temp=weather_data['temp']
            )
            today = datetime.datetime.now().replace(hour=9, minute=0, second=0, microsecond=0)
            add_event_to_calendar(f"Weather in {weather_data['city']}: {weather_data['description']}", today, description=f"Temperature: {weather_data['temp']:.1f}°C")
        else:
            response = get_response("get_weather_current_error")
    elif location_name:
        print(f"Fetching weather for {location_name}...")
        await text_to_speech_async(f"Fetching weather for {location_name}...")
        weather_data = await get_weather_async(location_name)
        if weather_data:
            response = get_response(
                "get_weather_city",
                city=weather_data['city'],
                description=weather_data['description'],
                temp=weather_data['temp']
            )
        else:
            response = get_response("get_weather_city_error", location=location_name)
    if not response:
        response = get_response("get_weather_unsure")
    await text_to_speech_async(response)
    return response


@intent_handler("add_calendar_event")
async def handle_add_calendar_event_intent(normalized_transcription: str) -> str:
    import dateparser
    response = ""
    patterns = [
        r"add (.+?)(?:\s+on|\s+at|\s+for)\s+(.+)",
        r"schedule (.+?)(?:\s+on|\s+at|\s+for)\s+(.+)",
        r"put (.+?)(?:\s+on my calendar|\s+in my calendar)(?:\s+for|\s+on|\s+at)\s+(.+)"
    ]
    summary = None
    date_str = None
    for pattern in patterns:
        match = re.search(pattern, normalized_transcription, re.IGNORECASE)
        if match:
            summary = match.group(1).strip()
            called_match = re.search(r"(.+?)\s+called\s+(.+)", summary, re.IGNORECASE)
            if called_match:
                 summary = f"{called_match.group(1).strip()}: {called_match.group(2).strip()}"
            date_str = match.group(2).strip()
            break
    if summary and date_str:
        start = dateparser.parse(date_str, settings={'PREFER_DATES_FROM': 'future'})
        if start:
            calendar_response = add_event_to_calendar(summary, start)
            response = get_response("add_calendar_event_success", calendar_response=calendar_response)
        else:
            response = get_response("add_calendar_event_parse_error")
    else:
        response = get_response("add_calendar_event_missing")
    await text_to_speech_async(response)
    return response


async def handle_interaction():
    try:
        greeting = get_response("greeting")
        print(f"Assistant (speaking): {greeting}")
        await text_to_speech_async(greeting)
        audio_data = await record_audio_async()
        transcription = await transcribe_audio_async(audio_data)
        if not transcription or not transcription.strip():
            print("No speech detected after greeting.")
            await text_to_speech_async(get_response("no_speech_detected"))
            return
        print(f"User said: {transcription}")
        await process_command(transcription)
    except Exception as e:
        print(f"[ERROR] Exception in handle_interaction: {e}")


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

                wake_event = asyncio.Event()

                def on_wakeword_detected():
                    if loop.is_running():
                        loop.call_soon_threadsafe(wake_event.set)

                wakeword_task = asyncio.create_task(run_wakeword_async(callback=on_wakeword_detected))
                await wake_event.wait()
                wakeword_task.cancel()
                try:
                    await wakeword_task
                except asyncio.CancelledError:
                    pass
                await handle_interaction()

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

# --- Auto-generated handlers for orphaned responses ---

@intent_handler("set_reminder_error")
async def handle_set_reminder_error(normalized_transcription: str) -> str:
    response = get_response("set_reminder_error")
    await text_to_speech_async(response)
    return response

@intent_handler("list_reminders_error")
async def handle_list_reminders_error(normalized_transcription: str) -> str:
    response = get_response("list_reminders_error")
    await text_to_speech_async(response)
    return response

@intent_handler("list_reminders_none")
async def handle_list_reminders_none(normalized_transcription: str) -> str:
    import datetime
    today = datetime.datetime.now().strftime('%A, %B %d, %Y')
    response = get_response("list_reminders_none", date=today)
    await text_to_speech_async(response)
    return response

@intent_handler("get_weather_current")
async def handle_get_weather_current(normalized_transcription: str) -> str:
    weather = await get_weather_async(None)
    if weather:
        response = get_response(
            "get_weather_current",
            city=weather['city'],
            description=weather['description'],
            temp=weather['temp']
        )
    else:
        response = get_response("get_weather_current_error")
    await text_to_speech_async(response)
    return response

@intent_handler("get_weather_city")
async def handle_get_weather_city(normalized_transcription: str) -> str:
    import re
    match = re.search(r'in ([A-Za-z\s]+)', normalized_transcription)
    city = match.group(1).strip() if match else None
    if city:
        weather = await get_weather_async(city)
        if weather:
            response = get_response(
                "get_weather_city",
                city=weather['city'],
                description=weather['description'],
                temp=weather['temp']
            )
        else:
            response = get_response("get_weather_city_error", location=city)
    else:
        response = get_response("get_weather_location_prompt")
    await text_to_speech_async(response)
    return response

@intent_handler("get_weather_current_error")
async def handle_get_weather_current_error(normalized_transcription: str) -> str:
    response = get_response("get_weather_current_error")
    await text_to_speech_async(response)
    return response

@intent_handler("get_weather_city_error")
async def handle_get_weather_city_error(normalized_transcription: str) -> str:
    import re
    match = re.search(r'in ([A-Za-z\s]+)', normalized_transcription)
    city = match.group(1).strip() if match else None
    response = get_response("get_weather_city_error", location=city or "the specified city")
    await text_to_speech_async(response)
    return response

@intent_handler("get_weather_unsure")
async def handle_get_weather_unsure(normalized_transcription: str) -> str:
    response = get_response("get_weather_unsure")
    await text_to_speech_async(response)
    return response

@intent_handler("get_weather_location_prompt")
async def handle_get_weather_location_prompt(normalized_transcription: str) -> str:
    response = get_response("get_weather_location_prompt")
    await text_to_speech_async(response)
    return response

@intent_handler("add_calendar_event_success")
async def handle_add_calendar_event_success(normalized_transcription: str) -> str:
    response = get_response("add_calendar_event_success", calendar_response="Event added to your calendar.")
    await text_to_speech_async(response)
    return response

@intent_handler("add_calendar_event_parse_error")
async def handle_add_calendar_event_parse_error(normalized_transcription: str) -> str:
    response = get_response("add_calendar_event_parse_error")
    await text_to_speech_async(response)
    return response

@intent_handler("add_calendar_event_missing")
async def handle_add_calendar_event_missing(normalized_transcription: str) -> str:
    response = get_response("add_calendar_event_missing")
    await text_to_speech_async(response)
    return response

@intent_handler("llm_service_error")
async def handle_llm_service_error(normalized_transcription: str) -> str:
    response = get_response("llm_service_error")
    await text_to_speech_async(response)
    return response

@intent_handler("llm_fallback_sorry")
async def handle_llm_fallback_sorry(normalized_transcription: str) -> str:
    response = get_response("llm_fallback_sorry")
    await text_to_speech_async(response)
    return response

@intent_handler("retrain_model_error")
async def handle_retrain_model_error(normalized_transcription: str) -> str:
    response = get_response("retrain_model_error", error="An error occurred during retraining.")
    await text_to_speech_async(response)
    return response

@intent_handler("no_speech_detected")
async def handle_no_speech_detected(normalized_transcription: str) -> str:
    response = get_response("no_speech_detected")
    await text_to_speech_async(response)
    return response
