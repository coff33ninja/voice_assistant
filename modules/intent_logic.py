import asyncio
import re
import datetime
from typing import Optional, Callable, Dict, Awaitable, Any
import pandas as pd
import json # For saving conversation
from datetime import datetime as dt # For timestamp in filename
import os
import threading
import logging # For logging warnings
import dateparser # Moved to top-level
# import sys # No longer needed for sys.exit

# Service function imports (these will call functions that use initialized services)
from modules.audio_utils import record_audio_async # For dedicated chat loop
from modules.stt_service import transcribe_audio_async # Import the transcription function
from modules.tts_service import text_to_speech_async
from modules.weather_service import get_weather_async
from modules.llm_service import get_llm_response
from modules.intent_classifier import detect_intent_async # type: ignore
from modules.reminder_utils import parse_reminder, parse_list_reminder_request
from modules.db_manager import save_reminder_async, get_reminders_for_date_async
from modules.gui_utils import show_reminders_gui # type: ignore
from modules.contractions import normalize_text # Ensure this is imported
from modules.error_handling import async_error_handler
from modules.calendar_utils import add_event_to_calendar

from modules.config import INTENT_RESPONSES_CSV # Import the missing variable
# Import for validation and retraining logic
from scripts.intent_validator import run_validation_and_retrain_async
from modules.retrain_utils import parse_retrain_request

# Configure a logger for this module (optional, or use root logger)
logger = logging.getLogger(__name__)
_PROJECT_ROOT_INTENT_LOGIC = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# --- Custom Exceptions ---
class ShutdownSignal(Exception):
    """Custom exception to signal graceful shutdown."""
    pass

# --- Intent Handling Registry ---
INTENT_HANDLERS: Dict[str, Callable[[str, Dict[str, Any]], Awaitable[str]]] = {}


def intent_handler(intent_name: str):

    def decorator(func: Callable[[str, Dict[str, Any]], Awaitable[str]]):
        INTENT_HANDLERS[intent_name] = func
        return func

    return decorator


# Load responses from CSV
# RESPONSES_PATH is no longer needed as INTENT_RESPONSES_CSV from config is used.
_responses_df = pd.read_csv(INTENT_RESPONSES_CSV) # Use imported path from config
RESPONSE_MAP = dict(zip(_responses_df["intent"], _responses_df["response"]))


def get_response(intent_key, **kwargs):
    resp = RESPONSE_MAP.get(intent_key, "")
    if resp and kwargs:
        try:
            return resp.format(**kwargs)
        except KeyError as e:
            logger.warning(
                "Missing key %s in response format for intent '%s'. Response: '%s', Kwargs: %s",
                e, intent_key, resp, kwargs
            )
            return resp  # Return unformatted response as fallback
        except Exception as e:  # Catch other formatting errors
            logger.warning(
                "Error formatting response for intent '%s': %s. Response: '%s', Kwargs: %s",
                intent_key, e, resp, kwargs
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
    logger.info(f"Assistant (greeting): {response}")
    await text_to_speech_async(response)
    return response


@intent_handler("goodbye")
async def handle_goodbye_intent(
    normalized_transcription: str, entities: Dict[str, Any]
) -> str:
    response = get_response("goodbye")
    logger.info(f"Assistant (goodbye): {response}")
    await text_to_speech_async(response)

    shutdown_message = "Shutting down assistant as requested by user."
    logger.info(shutdown_message)
    await text_to_speech_async(shutdown_message)
    await asyncio.sleep(0.2) # Small delay to allow TTS buffer to play

    raise ShutdownSignal("User requested shutdown.")


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
    logger.info(f"Retrain model message: {retrain_msg}") # Changed from print
    await text_to_speech_async(retrain_msg)
    return response # Return the initial response, retrain_msg is for spoken feedback


@async_error_handler()
async def process_command(transcription: str):
    normalized_transcription = normalize_text(transcription)
    logger.info(f"Processing command: {normalized_transcription}")

    intent, extracted_entities = await detect_intent_async(normalized_transcription)
    logger.info(f"Detected intent: {intent}, Entities: {extracted_entities}")
    
        # --- Heuristic check for "goodbye" --- Moved after the NLU
    goodbye_phrases = ["goodbye", "bye", "see you later", "farewell", "exit", "quit", "terminate"]
    if any(phrase in normalized_transcription.lower() for phrase in goodbye_phrases):
        logger.info("Heuristic match for 'goodbye' intent.")
        await handle_goodbye_intent(normalized_transcription, {})  # Call handler directly
        return  # Exit early as goodbye is handled

    # Special handling for retrain_model as it's combined with parse_retrain_request
    if intent == "retrain_model" or parse_retrain_request(normalized_transcription):
        response = get_response("retrain_model")
        await text_to_speech_async(response)
        try:
            _success, retrain_msg = await run_validation_and_retrain_async()
        except Exception as e:
            retrain_msg = get_response("retrain_model_error", error=str(e))
        logger.info(f"Retrain message: {retrain_msg}")
        await text_to_speech_async(retrain_msg)
        return  # Exit early as speech is handled

    handler = INTENT_HANDLERS.get(intent)  # Check registered handlers first
    response_text = ""  # Initialize for clarity, though handlers return their own

    if handler:
        response_text = await handler(normalized_transcription, extracted_entities)
    else:  # Fallback to LLM
        logger.info("Sending to LLM for general query or unhandled/low-confidence intent...")
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
        logger.info(f"Assistant (LLM): {response_text}")
        if response_text: # Ensure LLM response is not empty before speaking
            await text_to_speech_async(response_text)
    # Individual handlers (including LLM path) now manage their own TTS.
    # The 'response_text' variable here primarily holds what was spoken for logging/debugging if needed.


# Define placeholder handlers for intents previously in the first process_command
@intent_handler("set_reminder")
async def handle_set_reminder_intent(
    normalized_transcription: str, entities: Dict[str, Any]
) -> str:
    task = entities.get("task")
    time_phrase = entities.get("time_phrase")
    reminder_time_obj = None

    if time_phrase and dateparser:
        # Try parsing the time_phrase using dateparser
        reminder_time_obj = dateparser.parse(
            time_phrase, settings={"PREFER_DATES_FROM": "future"}
        )

    if (
        not task or not reminder_time_obj
    ):  # Fallback to full parsing if entities are insufficient
        logger.info(
            "Entities for set_reminder not fully resolved or missing, falling back to full parse_reminder with entities."
        )
        # Pass entities to parse_reminder, which will use them or fall back to its own regex
        parsed_reminder_data = parse_reminder(normalized_transcription, entities=entities)
        if parsed_reminder_data:
            task = parsed_reminder_data["task"] # Use results from parse_reminder
            reminder_time_obj = parsed_reminder_data["time"]
        else:
            response_to_speak = get_response("set_reminder_error")
            await text_to_speech_async(response_to_speak)
            return response_to_speak

    response_parts = []
    if task and reminder_time_obj:
        await save_reminder_async(task, reminder_time_obj)
        add_event_to_calendar(task, reminder_time_obj)  # Add reminder to calendar
        response = get_response(
            "reminder_set_full",
            task=task,
            time=reminder_time_obj.strftime("%I:%M %p on %A, %B %d"),
        )
        response_parts.append(response)
        response_parts.append(
            get_response("reminder_added_to_calendar")
        )
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
        response_parts.append(response)

    response_to_speak = " ".join(filter(None, response_parts))
    if not response_to_speak:
        response_to_speak = get_response("set_reminder_success_generic")
    await text_to_speech_async(response_to_speak)
    return response_to_speak


@intent_handler("list_reminders")
async def handle_list_reminders_intent(
    normalized_transcription: str, entities: Dict[str, Any]
) -> str:
    # date_reference = entities.get("date_reference") # This is now handled inside parse_list_reminder_request
    # The first argument to parse_list_reminder_request is the full text,
    # and entities are passed as a keyword argument.
    target_date = parse_list_reminder_request(normalized_transcription, entities=entities)
    response = ""
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
    location_entity = entities.get("location")
    use_current_location = False
    response = ""
    location_name_to_fetch: Optional[str] = None

    if location_entity:
        if location_entity.lower() in ["current", "my area", "here"]:
            use_current_location = True
        else:
            location_name_to_fetch = location_entity
    else:
        logger.info("No location entity from NLU for get_weather, attempting regex.")
        my_area_phrases = ["my area", "here", "current location", "around me", "local weather"]
        simple_weather_queries = ["what's the weather", "weather today", "weather now", "tell me the weather", "weather"]
        location_match = re.search(r"(?:weather in|weather for|weather at|weather like in)\s+([A-Za-z\s]+)", normalized_transcription.lower())
        extracted_location_regex = location_match.group(1).strip() if location_match else None
        if extracted_location_regex:
            if extracted_location_regex.lower() in [p.lower() for p in my_area_phrases]:
                use_current_location = True
            else:
                location_name_to_fetch = extracted_location_regex
        else:
            transcription_lower_stripped = normalized_transcription.lower().strip()
            is_simple_query = transcription_lower_stripped in simple_weather_queries
            is_my_area_query = any(phrase in transcription_lower_stripped for phrase in my_area_phrases)
            if is_my_area_query or is_simple_query:
                use_current_location = True

    if not use_current_location and not location_name_to_fetch:
        response = get_response("get_weather_location_prompt")
        await text_to_speech_async(response)
        return response

    weather_data = None
    if use_current_location:
        logger.info("Fetching weather for current location (entities will be passed)...")
        await text_to_speech_async("Fetching weather for current location...")
        # Pass entities to get_weather_async
        weather_data = await get_weather_async(location_query=None, entities=entities)
        if weather_data and "error" not in weather_data and "message" not in weather_data :
            response = get_response("get_weather_current", city=weather_data["city"], description=weather_data["description"], temp=weather_data["temp"])
            today = datetime.datetime.now().replace(hour=9, minute=0, second=0, microsecond=0)
        elif weather_data and "message" in weather_data: # Forecast message
            response = weather_data["message"]
        elif weather_data and "error" in weather_data: # Error from service
            response = weather_data["error"]
            add_event_to_calendar(f"Weather in {weather_data['city']}: {weather_data['description']}", today, description=f"Temperature: {weather_data['temp']:.1f}°C")
        else: # pragma: no cover
            response = get_response("get_weather_current_error")
    elif location_name_to_fetch:
        logger.info(f"Fetching weather for {location_name_to_fetch} (entities will be passed)...")
        await text_to_speech_async(f"Fetching weather for {location_name_to_fetch}...")
        # Pass entities to get_weather_async
        weather_data = await get_weather_async(location_query=location_name_to_fetch, entities=entities)
        if weather_data and "error" not in weather_data and "message" not in weather_data:
            response = get_response("get_weather_city", city=weather_data["city"], description=weather_data["description"], temp=weather_data["temp"])
        elif weather_data and "message" in weather_data: # Forecast message
            response = weather_data["message"]
        elif weather_data and "error" in weather_data: # Error from service
            response = weather_data["error"]
        else: # General error if weather_data is None or unexpected
            response = get_response("get_weather_city_error", location=location_name_to_fetch)

    if not response: # Fallback if no specific weather response (e.g. all conditions above failed)
        response = get_response("get_weather_unsure")

    await text_to_speech_async(response)
    return response


@intent_handler("start_chat_with_llm")
async def handle_start_chat_with_llm(
    normalized_transcription: str, entities: Dict[str, Any]
) -> str:
    initial_response = get_response("start_chat_with_llm")
    await text_to_speech_async(initial_response)

    conversation_log_for_saving = []
    stop_phrase = "stop chat and save"

    while True:
        try:
            # Use a slightly longer recording duration for more natural conversation flow
            audio_data = await asyncio.wait_for(record_audio_async(duration=7), timeout=10.0)
        except asyncio.TimeoutError:
            logger.warning("Chat mode: Audio recording timed out.")
            await text_to_speech_async("I didn't hear anything. Are you still there? Say 'stop chat and save' to end.")
            continue
        except Exception as rec_e:
            logger.error(f"Chat mode: Error during audio recording: {rec_e}")
            await text_to_speech_async("Sorry, there was an issue with audio recording during our chat.")
            break # Exit chat on recording error

        if audio_data is None or not audio_data.any():
            logger.info("Chat mode: No audio data captured.")
            # Don't say anything, just loop back to listen
            continue

        user_input = await transcribe_audio_async(audio_data)
        if not user_input or not user_input.strip():
            logger.info("Chat mode: No speech detected from user.")
            # Don't say anything, just loop back to listen
            continue

        logger.info(f"Chat mode - User said: {user_input}")

        if stop_phrase in user_input.lower():
            logger.info("Chat mode: Stop phrase detected. Ending chat session.")
            break

        llm_response_text = await get_llm_response(user_input)
        if not llm_response_text or llm_response_text.strip() == "":
            llm_response_text = "The AI didn't provide a response for that. You can try rephrasing or say 'stop chat and save'."
            logger.warning("Chat mode: LLM returned empty response.")

        logger.info(f"Chat mode - Assistant (LLM): {llm_response_text}")
        await text_to_speech_async(llm_response_text)
        conversation_log_for_saving.append({"user": user_input, "assistant": llm_response_text})

    # After loop (stop phrase or error)
    if conversation_log_for_saving:
        conversations_dir = os.path.join(_PROJECT_ROOT_INTENT_LOGIC, "conversations")
        os.makedirs(conversations_dir, exist_ok=True)
        timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(conversations_dir, f"chat_session_{timestamp}.json")
        try:
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(conversation_log_for_saving, f, indent=2, ensure_ascii=False)
            final_response = get_response("chat_session_saved")
        except Exception as e:
            logger.error(f"Chat mode: Error saving conversation to {filename}: {e}")
            final_response = get_response("chat_session_save_error")
    else:
        final_response = get_response("chat_session_empty")

    await text_to_speech_async(final_response)
    return final_response # Return the final status message


@intent_handler("add_calendar_event")
async def handle_add_calendar_event_intent(
    normalized_transcription: str, entities: Dict[str, Any]
) -> str:
    response = ""
    summary = None
    start_time_obj = None
    used_entities = False

    # Try to get info from entities first
    entity_summary = entities.get("summary") or entities.get("task") # Common entity keys for event description
    entity_time_phrase = entities.get("time_phrase") or entities.get("event_datetime_str") # Common entity keys for time

    if entity_summary and entity_time_phrase and dateparser:
        logger.info(f"Attempting to parse calendar event from entities: summary='{entity_summary}', time_phrase='{entity_time_phrase}'")
        start_time_obj = dateparser.parse(entity_time_phrase, settings={"PREFER_DATES_FROM": "future"})
        if start_time_obj:
            summary = entity_summary
            used_entities = True
            logger.info(f"Successfully parsed event from entities: summary='{summary}', start_time='{start_time_obj}'")
        else:
            logger.warning(f"Failed to parse time_phrase '{entity_time_phrase}' from entities.")

    if not summary or not start_time_obj: # If entities didn't provide full info or parsing failed
        if used_entities: # Log if we tried entities but they were insufficient
            logger.info("Entities were present but insufficient for calendar event, falling back to regex.")
        else: # Log if entities were not even present for summary/time
            logger.info("No suitable entities for calendar event summary/time, falling back to regex.")

        patterns = [
            r"add (.+?)(?:\s+on|\s+at|\s+for)\s+(.+)",
            r"schedule (.+?)(?:\s+on|\s+at|\s+for)\s+(.+)",
            r"put (.+?)(?:\s+on my calendar|\s+in my calendar)(?:\s+for|\s+on|\s+at)\s+(.+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, normalized_transcription, re.IGNORECASE)
            if match:
                summary = match.group(1).strip()
                called_match = re.search(r"(.+?)\s+called\s+(.+)", summary, re.IGNORECASE)
                if called_match:
                    summary = f"{called_match.group(1).strip()}: {called_match.group(2).strip()}"
                date_str_regex = match.group(2).strip()
                if dateparser:
                    start_time_obj = dateparser.parse(date_str_regex, settings={"PREFER_DATES_FROM": "future"})
                break

    if summary and start_time_obj:
        calendar_response = add_event_to_calendar(summary, start_time_obj)
        response = get_response("add_calendar_event_success", calendar_response=calendar_response)
    elif summary and not start_time_obj: # Date parsing failed or was not provided
        response = get_response("add_calendar_event_parse_error")
    else: # Summary or date missing
        response = get_response("add_calendar_event_missing")

    await text_to_speech_async(response)
    return response
