import threading
import asyncio
import re
import platform
import datetime  # Added for global datetime access
from typing import Optional
import warnings
import logging

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
from modules.contractions import normalize_text
from modules.error_handling import async_error_handler
from typing import Callable, Dict, Awaitable, Any
from modules.calendar_utils import (
    add_event_to_calendar,
)  # get_calendar_file_path not used directly here yet

# --- Modularized interaction logic ---

# Import for validation and retraining logic
from scripts.intent_validator import run_validation_and_retrain_async
from modules.retrain_utils import parse_retrain_request  # Still need this for parsing

# --- Suppress specific library warnings and configure logging ---

# 1. SpeechBrain UserWarning for 'speechbrain.pretrained'
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r"Module 'speechbrain.pretrained' was deprecated.*", # Use raw string for regex
)

# 2. TTS (torch.load) FutureWarning
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r"You are using `torch\.load` with `weights_only=False`.*" # Use raw string for regex
)

# 3. WhisperX / Pyannote version mismatch warnings (these are often UserWarning)
# These messages are printed by WhisperX itself, not necessarily through pyannote.audio's logger.
warnings.filterwarnings(
    "ignore",
    # category=UserWarning, # Can be specified if known, otherwise matches any category
    message=r"Model was trained with pyannote\.audio .* yours is .*" # Use raw string for regex
)
warnings.filterwarnings(
    "ignore",
    # category=UserWarning,
    message=r"Model was trained with torch .* yours is .*" # Use raw string for regex
)

# Configure logging levels for verbose libraries
logging.getLogger("TTS").setLevel(
    logging.WARNING
)  # Suppress Coqui TTS INFO messages (e.g., config printout)
logging.getLogger("pyannote.audio").setLevel(
    logging.ERROR
)  # Suppress pyannote.audio warnings (e.g., version mismatch)

# --- Intent Handling Registry ---
INTENT_HANDLERS: Dict[str, Callable[[str, Dict[str, Any]], Awaitable[str]]] = {}


def intent_handler(intent_name: str):

    def decorator(func: Callable[[str, Dict[str, Any]], Awaitable[str]]):
        INTENT_HANDLERS[intent_name] = func
        return func

    return decorator


# Load responses from CSV
RESPONSES_PATH = os.path.join(
    os.path.dirname(__file__), "intent_data", "intent_responses.csv"
)
_responses_df = pd.read_csv(RESPONSES_PATH)
RESPONSE_MAP = dict(zip(_responses_df["intent"], _responses_df["response"]))


def get_response(intent_key, **kwargs):
    resp = RESPONSE_MAP.get(intent_key, "")
    if resp and kwargs:
        try:
            return resp.format(**kwargs)
        except KeyError as e:
            print(
                f"Warning: Missing key {e} in response format for intent '{intent_key}'. Response: '{resp}', Kwargs: {kwargs}"
            )
            return resp  # Return unformatted response as fallback
        except Exception as e:  # Catch other formatting errors
            print(
                f"Warning: Error formatting response for intent '{intent_key}': {e}. Response: '{resp}', Kwargs: {kwargs}"
            )
            return resp
    return resp


@intent_handler("cancel_task")
async def handle_cancel_task(
    normalized_transcription: str, entities: Dict[str, Any]
) -> str:
    response = get_response("cancel_task")
    await text_to_speech_async(response)
    return response


@intent_handler("calendar_query")
async def handle_calendar_query(
    normalized_transcription: str, entities: Dict[str, Any]
) -> str:
    response = get_response("calendar_query")
    await text_to_speech_async(response)
    return response


@intent_handler("greeting")
async def handle_greeting_intent(
    normalized_transcription: str, entities: Dict[str, Any]
) -> str:
    response = get_response("greeting")
    print(f"Assistant (greeting): {response}")
    await text_to_speech_async(response)
    return response


@intent_handler("goodbye")
async def handle_goodbye_intent(
    normalized_transcription: str, entities: Dict[str, Any]
) -> str:
    response = get_response("goodbye")
    print(f"Assistant (goodbye): {response}")
    await text_to_speech_async(response)
    print("Shutting down assistant as requested by user.")
    await text_to_speech_async("Shutting down assistant as requested by user.")
    import sys

    sys.exit(0)


@intent_handler("retrain_model")
async def handle_retrain_model_intent(
    normalized_transcription: str, entities: Dict[str, Any]
) -> str:
    response = get_response("retrain_model")
    await text_to_speech_async(response)
    try:
        _success, retrain_msg = await run_validation_and_retrain_async()
    except Exception as e:
        retrain_msg = get_response("retrain_model_error", error=str(e))
    print(retrain_msg)
    await text_to_speech_async(retrain_msg)
    return response


@async_error_handler()
async def process_command(transcription: str):
    normalized_transcription = normalize_text(transcription)
    print(f"Processing command: {normalized_transcription}")

    intent, extracted_entities = await detect_intent_async(normalized_transcription)

    # Special handling for retrain_model as it's combined with parse_retrain_request
    if intent == "retrain_model" or parse_retrain_request(normalized_transcription):
        response = get_response("retrain_model")
        await text_to_speech_async(response)
        try:
            _success, retrain_msg = await run_validation_and_retrain_async()
        except Exception as e:
            retrain_msg = get_response("retrain_model_error", error=str(e))
        print(retrain_msg)
        await text_to_speech_async(retrain_msg)
        return  # Exit early as speech is handled

    handler = INTENT_HANDLERS.get(intent)  # Check registered handlers first
    response_text = ""  # Initialize for clarity, though handlers return their own

    if handler:
        response_text = await handler(normalized_transcription, extracted_entities)
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
async def handle_set_reminder_intent(
    normalized_transcription: str, entities: Dict[str, Any]
) -> str:
    import dateparser  # Moved import here as it's specific to this handler

    task = entities.get("task")
    time_phrase = entities.get("time_phrase")
    reminder_time_obj = None

    if time_phrase:
        # Try parsing the time_phrase using dateparser
        reminder_time_obj = dateparser.parse(
            time_phrase, settings={"PREFER_DATES_FROM": "future"}
        )

    if (
        not task or not reminder_time_obj
    ):  # Fallback to full parsing if entities are insufficient
        print(
            "Entities for reminder not fully resolved or missing, falling back to parse_reminder."
        )
        parsed_reminder_data = parse_reminder(normalized_transcription)
        if parsed_reminder_data:
            task = parsed_reminder_data["task"]
            reminder_time_obj = parsed_reminder_data["time"]
        else:
            response_to_speak = get_response("set_reminder_error")
            await text_to_speech_async(response_to_speak)
            return response_to_speak

    response_parts = []  # Initialize the list here
    if task and reminder_time_obj:
        await save_reminder_async(task, reminder_time_obj)
        add_event_to_calendar(task, reminder_time_obj)  # Add reminder to calendar
        response = get_response(
            "reminder_set_full",  # Use granular key
            task=task,
            time=reminder_time_obj.strftime("%I:%M %p on %A, %B %d"),
        )
        response_parts.append(response)
        response_parts.append(
            get_response("reminder_added_to_calendar")
        )  # Add calendar confirmation part
    elif task and not reminder_time_obj:
        response_parts.append(get_response("reminder_set_task_only", task=task))
        response_parts.append(get_response("reminder_ask_for_time"))
    elif not task and reminder_time_obj:
        response_parts.append(
            get_response(
                "reminder_set_time_only",
                time=reminder_time_obj.strftime("%I:%M %p on %A, %B %d"),
            )
        )
        response_parts.append(get_response("reminder_ask_for_task"))
    else:
        response = get_response("set_reminder_error")
        response_parts.append(response)  # Add the error response part

    response_to_speak = " ".join(filter(None, response_parts))
    if not response_to_speak:  # Fallback if no parts were added somehow
        response_to_speak = get_response(
            "set_reminder_success_generic"
        )  # Use generic success key
    await text_to_speech_async(response_to_speak)
    return response_to_speak


@intent_handler("list_reminders")
async def handle_list_reminders_intent(
    normalized_transcription: str, entities: Dict[str, Any]
) -> str:
    date_reference = entities.get("date_reference")
    target_date = parse_list_reminder_request(
        date_reference or normalized_transcription or ""
    )
    if target_date:
        reminders_found = await get_reminders_for_date_async(target_date)
        date_str = target_date.strftime("%A, %B %d, %Y")
        if reminders_found:
            reminders_text = " ".join(
                f"{r['task']} at {r['time'].strftime('%I:%M %p')}."
                for r in reminders_found
            )
            response = get_response(
                "list_reminders", date=date_str, reminders=reminders_text
            )
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
async def handle_get_weather_intent(
    normalized_transcription: str, entities: Dict[str, Any]
) -> str:
    location_entity = entities.get("location")  # This could be a city name or "current"
    use_current_location = False
    response = ""
    location_name_to_fetch: Optional[str] = None

    if location_entity:
        if location_entity.lower() in ["current", "my area", "here"]:
            use_current_location = True
        else:
            location_name_to_fetch = location_entity
    else:  # Fallback to regex if no location entity from NLU
        print("No location entity from NLU, attempting regex for weather.")
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
        extracted_location_regex = (
            location_match.group(1).strip() if location_match else None
        )
        if extracted_location_regex:
            if extracted_location_regex.lower() in [p.lower() for p in my_area_phrases]:
                use_current_location = True
            else:
                location_name_to_fetch = extracted_location_regex
        else:
            transcription_lower_stripped = normalized_transcription.lower().strip()
            is_simple_query = transcription_lower_stripped in simple_weather_queries
            is_my_area_query = any(
                phrase in transcription_lower_stripped for phrase in my_area_phrases
            )
            if is_my_area_query or is_simple_query:
                use_current_location = True

    if not use_current_location and not location_name_to_fetch:
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
                city=weather_data["city"],
                description=weather_data["description"],
                temp=weather_data["temp"],
            )
            today = datetime.datetime.now().replace(
                hour=9, minute=0, second=0, microsecond=0
            )
            add_event_to_calendar(
                f"Weather in {weather_data['city']}: {weather_data['description']}",
                today,
                description=f"Temperature: {weather_data['temp']:.1f}°C",
            )
        else:
            response = get_response("get_weather_current_error")
    elif location_name_to_fetch:
        print(f"Fetching weather for {location_name_to_fetch}...")
        await text_to_speech_async(f"Fetching weather for {location_name_to_fetch}...")
        weather_data = await get_weather_async(location_name_to_fetch)
        if weather_data:
            response = get_response(
                "get_weather_city",
                city=weather_data["city"],
                description=weather_data["description"],
                temp=weather_data["temp"],
            )
        else:
            response = get_response(
                "get_weather_city_error", location=location_name_to_fetch
            )
    if not response:
        response = get_response("get_weather_unsure")
    await text_to_speech_async(response)
    return response


@intent_handler("add_calendar_event")
async def handle_add_calendar_event_intent(
    normalized_transcription: str, entities: Dict[str, Any]
) -> str:
    import dateparser  # Moved import here

    response = ""
    summary = entities.get("event_summary")
    date_str = entities.get("event_datetime_str")  # Expecting NLU to provide this
    start_time_obj = None

    if date_str:
        start_time_obj = dateparser.parse(
            date_str, settings={"PREFER_DATES_FROM": "future"}
        )

    if (
        not summary or not start_time_obj
    ):  # Fallback to regex if entities are insufficient
        print(
            "Entities for calendar event not fully resolved or missing, falling back to regex."
        )
        patterns = [
            r"add (.+?)(?:\s+on|\s+at|\s+for)\s+(.+)",
            r"schedule (.+?)(?:\s+on|\s+at|\s+for)\s+(.+)",
            r"put (.+?)(?:\s+on my calendar|\s+in my calendar)(?:\s+for|\s+on|\s+at)\s+(.+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, normalized_transcription, re.IGNORECASE)
            if match:
                summary = match.group(1).strip()
                # Handle "called" construct within summary if NLU didn't separate it
                called_match = re.search(
                    r"(.+?)\s+called\s+(.+)", summary, re.IGNORECASE
                )
                if called_match:
                    summary = f"{called_match.group(1).strip()}: {called_match.group(2).strip()}"
                date_str_regex = match.group(2).strip()
                start_time_obj = dateparser.parse(
                    date_str_regex, settings={"PREFER_DATES_FROM": "future"}
                )
                break

    if summary and start_time_obj:  # Check start_time_obj instead of date_str
        if start_time_obj:
            calendar_response = add_event_to_calendar(summary, start_time_obj)
            response = get_response(
                "add_calendar_event_success", calendar_response=calendar_response
            )
        else:
            response = get_response("add_calendar_event_parse_error")
    else:
        response = get_response("add_calendar_event_missing")
    await text_to_speech_async(response)
    return response


# Helper function to get a descriptive name for a task
def task_name(task: asyncio.Task) -> str:
    try:
        return task.get_name()  # Python 3.8+
    except AttributeError:
        return str(task)


async def handle_interaction():
    try:
        greeting = get_response("greeting")
        print(f"Assistant (speaking): {greeting}")
        await text_to_speech_async(greeting)

        audio_data = None
        try:
            # Record audio with a 10-second timeout
            audio_data = await asyncio.wait_for(record_audio_async(), timeout=10.0)
        except asyncio.TimeoutError:
            print("Audio recording timed out.")
            await text_to_speech_async("Sorry, I couldn't capture audio in time.")
            return
        except Exception as rec_e:
            print(f"Error during audio recording: {rec_e}")
            await text_to_speech_async(
                "Sorry, there was an issue with audio recording."
            )
            return

        if audio_data is None or not audio_data.any():  # Check if audio_data is valid
            print("No audio data captured or audio data is empty.")
            await text_to_speech_async(get_response("no_speech_detected"))
            return

        transcription = await transcribe_audio_async(audio_data)
        if not transcription or not transcription.strip():
            print("No speech detected after greeting.")
            await text_to_speech_async(get_response("no_speech_detected"))
            return
        print(f"User said: {transcription}")
        await process_command(transcription)
    except Exception as e:
        print(f"[ERROR] Exception in handle_interaction: {e}")


async def main_loop(loop: asyncio.AbstractEventLoop):
    # Keep track of tasks to cancel them on exit
    background_tasks = []
    reminder_task = loop.create_task(reminder_check_loop(text_to_speech_async))
    background_tasks.append(reminder_task)

    current_wakeword_task: Optional[asyncio.Task] = None

    try:
        while True:
            print("Waiting for wake word...")
            wake_event = asyncio.Event()

            def on_wakeword_detected():
                if loop.is_running() and not wake_event.is_set():
                    loop.call_soon_threadsafe(wake_event.set)

            if current_wakeword_task and not current_wakeword_task.done():
                current_wakeword_task.cancel()

            current_wakeword_task = loop.create_task(
                run_wakeword_async(callback=on_wakeword_detected)
            )

            try:
                await wake_event.wait()
            except asyncio.CancelledError:
                print("Main loop's wait for wake_event cancelled.")
                raise

            if current_wakeword_task:
                current_wakeword_task.cancel()
                try:
                    await current_wakeword_task
                except asyncio.CancelledError:
                    pass

            await handle_interaction()
    except asyncio.CancelledError:
        print("Main loop was cancelled.")
    finally:
        print("Main loop ending. Cancelling background tasks...")
        if current_wakeword_task and not current_wakeword_task.done():
            current_wakeword_task.cancel()
            try:
                await current_wakeword_task
            except asyncio.CancelledError:
                print("Wakeword task cancelled during main loop cleanup.")

        for task in background_tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    print(f"Background task {task_name(task)} was cancelled.")
                except Exception as e:
                    print(
                        f"Error during cancellation of background task {task_name(task)}: {e}"
                    )
        print("All background tasks in main_loop processed for cancellation.")


def run_assistant():
    # --- Initialization ---
    print("Initializing services...")
    # For Windows, set the policy to allow more threads if needed
    if platform.system() == "Windows":
        import nest_asyncio

        nest_asyncio.apply()

    loop = asyncio.get_event_loop()  # Get the event loop

    initialize_stt()
    initialize_tts()
    initialize_weather_service()
    initialize_llm()
    initialize_intent_classifier()
    initialize_db()

    print("Services initialized. You can now interact with the assistant.")

    main_task: Optional[asyncio.Task] = None
    try:
        main_task = loop.create_task(main_loop(loop))
        loop.run_until_complete(main_task)
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received. Shutting down gracefully...")
        if main_task and not main_task.done():
            main_task.cancel()
            try:
                # Give main_task a chance to clean up
                loop.run_until_complete(main_task)
            except asyncio.CancelledError:
                print("Main task successfully cancelled.")
            except RuntimeError as e:  # Handle cases where loop might be closing
                print(f"Runtime error during main_task cleanup: {e}")
            except Exception as e:
                print(f"Exception during main_task cleanup: {e}")
    except Exception as e:  # Catch other unexpected errors from main_loop
        print(f"Unexpected error in run_assistant: {e}")
        if (
            main_task and not main_task.done()
        ):  # Attempt to cancel main_task if it's still running
            main_task.cancel()
            if loop.is_running():
                loop.run_until_complete(main_task)  # Allow it to process cancellation
    finally:
        print("Performing final cleanup of any remaining tasks...")
        remaining_tasks = [
            t
            for t in asyncio.all_tasks(loop=loop)
            if t is not asyncio.current_task(loop=loop) and not t.done()
        ]
        if remaining_tasks:
            print(f"Cancelling {len(remaining_tasks)} remaining tasks...")
            for task in remaining_tasks:
                task.cancel()
            if loop.is_running():
                loop.run_until_complete(
                    asyncio.gather(*remaining_tasks, return_exceptions=True)
                )

        print("Closing event loop.")
        if not loop.is_closed():
            loop.close()
        print("Assistant shut down.")


if __name__ == "__main__":
    run_assistant()
