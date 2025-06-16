import threading
import asyncio
import re
import platform
from typing import Optional

from wakeword_detector import run_wakeword

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

# --- Modularized interaction logic ---

# --- Intent Handling Registry ---
INTENT_HANDLERS: Dict[str, Callable[[str], Awaitable[str]]] = {}


def intent_handler(intent_name: str):
    def decorator(func: Callable[[str], Awaitable[str]]):
        INTENT_HANDLERS[intent_name] = func
        return func

    return decorator


async def handle_set_reminder(normalized_transcription: str) -> str:
    # Normalize contractions and pronunciation issues
    normalized_transcription = normalize_text(normalized_transcription)
    print(f"Processing command: {normalized_transcription}")
    await text_to_speech_async(f"Processing command: {normalized_transcription}")
    intent = await detect_intent_async(normalized_transcription)
    response = ""
    if intent == "set_reminder":
        reminder = parse_reminder(normalized_transcription)
        if reminder:
            await save_reminder_async(reminder["task"], reminder["time"])
            response = f"Okay, I've set a reminder for '{reminder['task']}' at {reminder['time'].strftime('%I:%M %p on %A, %B %d')}"
        else:
            response = "I couldn't quite understand the reminder. Please try saying something like 'remind me to call John tomorrow at 2 pm'."
        return response  # Added return
    elif intent == "list_reminders":
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
            # Show reminders in GUI and read out loud
            import threading

            threading.Thread(target=show_reminders_gui, args=(reminders_found, date_str), daemon=True).start()  # type: ignore
        else:
            response = "I couldn't understand which date you want reminders for. Please specify a day like 'today', 'tomorrow', or a specific date."
        return response  # Added return
    elif intent == "retrain_model" or parse_retrain_request(normalized_transcription):
        response = "Starting model retraining. This may take a few minutes."
        await text_to_speech_async(response)
        try:
            success, retrain_msg = await trigger_model_retraining_async()
            # The message from trigger_model_retraining_async is already comprehensive
        except Exception as e:
            retrain_msg = f"Retraining failed due to an error: {e}"

        print(retrain_msg)
        await text_to_speech_async(retrain_msg)
        return ""  # Return empty string as speech is handled
    elif intent == "get_weather":
        location_name: Optional[str] = None
        use_current_location = False

    # Define phrases that indicate current location
    my_area_phrases = [
        "my area",
        "here",
        "current location",
        "around me",
        "local weather",
    ]
    # Define simple queries that imply current location if no specific location is given
    simple_weather_queries = [
        "what's the weather",
        "weather today",
        "weather now",
        "tell me the weather",
        "weather",
    ]

    # Try to extract a specific location from the transcription
    location_match = re.search(
        r"(?:weather in|weather for|weather at|weather like in)\s+([A-Za-z\s]+)",
        normalized_transcription.lower(),
    )
    extracted_location = location_match.group(1).strip() if location_match else None

    if extracted_location:
        # Check if the extracted location is actually a "my area" phrase
        if extracted_location.lower() in [p.lower() for p in my_area_phrases]:
            use_current_location = True
        else:
            location_name = extracted_location
    else:
        # No specific location like "weather in X", check for general "my area" or simple queries
        transcription_lower_stripped = normalized_transcription.lower().strip()
        is_simple_query = transcription_lower_stripped in simple_weather_queries
        is_my_area_query = any(
            phrase in transcription_lower_stripped for phrase in my_area_phrases
        )

        if is_my_area_query or is_simple_query:
            use_current_location = True
        else:
            # If intent is get_weather but no clear location or "my area" phrase, prompt the user
            response = "Which location's weather are you interested in? For example, say 'what is the weather in London' or 'what is the weather in my area'."
            await text_to_speech_async(response)
            return ""  # Exit early as we need more information

    # Now, fetch weather based on determined location_name or use_current_location
    weather_data = None
    if use_current_location:
        print("Fetching weather for current location...")
        await text_to_speech_async("Fetching weather for current location...")
        weather_data = await get_weather_async(
            None
        )  # Pass None to signify current location
        if weather_data:
            response = f"The current weather in {weather_data['city']} is {weather_data['description']} with a temperature of {weather_data['temp']:.1f} degrees Celsius."

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

    if not response:  # Fallback if no specific response was generated
        response = "I'm not sure which location you're asking about for the weather. Please specify, like 'weather in London' or 'weather in my area'."
    await text_to_speech_async(response)
    return response  # Added return


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
    await text_to_speech_async(f"Processing command: {normalized_transcription}")
    intent = await detect_intent_async(normalized_transcription)

    # Special handling for retrain_model as it's combined with parse_retrain_request
    if intent == "retrain_model" or parse_retrain_request(normalized_transcription):
        response = "Starting model retraining. This may take a few minutes."
        await text_to_speech_async(response)
        try:
            success, retrain_msg = await trigger_model_retraining_async()
        except Exception as e:
            retrain_msg = f"Retraining failed due to an error: {e}"
        print(retrain_msg)
        await text_to_speech_async(retrain_msg)
        return  # Exit early as speech is handled

    handler = INTENT_HANDLERS.get(intent)  # Check registered handlers first

    if handler:
        response = await handler(normalized_transcription)
    elif (
        intent == "set_reminder"
    ):  # Direct call for now, can be refactored into handler
        response = await handle_set_reminder(normalized_transcription)
    elif intent == "list_reminders":  # Direct call for now
        response = await handle_list_reminders_intent(
            normalized_transcription
        )  # Call correct handler
    elif intent == "get_weather":  # Direct call for now
        response = await handle_get_weather_intent(
            normalized_transcription
        )  # Call correct handler
    else:  # Fallback to LLM
        print("Sending to LLM for general query or unhandled/low-confidence intent...")
        response = await get_llm_response(input_text=normalized_transcription)
        if (
            not response
            or "don't understand" in response.lower()
            or "sorry" in response.lower()
        ):  # Broader check for LLM uncertainty
            response = (
                "I'm sorry, I didn't understand that. "
                "You can ask me to set reminders, check the weather, or answer questions. "
                "Try rephrasing your request or say 'help' for examples."
            )

    if response:  # Ensure there's a response to speak
        print(f"Assistant: {response}")
        await text_to_speech_async(response)


# Define placeholder handlers for intents previously in the first process_command
@intent_handler("set_reminder")
async def handle_set_reminder_intent(normalized_transcription: str) -> str:
    reminder = parse_reminder(normalized_transcription)
    if reminder:
        response = f"Okay, I've set a reminder for '{reminder['task']}' at {reminder['time'].strftime('%I:%M %p on %A, %B %d')}"
        await save_reminder_async(reminder["task"], reminder["time"])
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
                    loop.call_soon_threadsafe(wake_event.set)

                run_wakeword(callback=on_wakeword_detected)
                await wake_event.wait()

                print("Wake word detected! Greeting user...")
                greeting = get_greeting()
                await text_to_speech_async(greeting)
                print("Greeting finished. Listening for command...")
                audio_data = await record_audio_async()
                transcription = await transcribe_audio_async(audio_data)
                if not transcription or not transcription.strip():
                    print("No speech detected after greeting.")
                    await text_to_speech_async("I didn't catch that. If you need something, please call me again.")
                    continue
                print(f"User said: {transcription}")
                await process_command(transcription)

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
