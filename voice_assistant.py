import threading
import asyncio
import re
import platform
from typing import Optional

from wakeword_detector import run_wakeword

from modules.config import GREETING_MESSAGE
from modules.audio_utils import record_audio_async
from modules.stt_service import initialize_stt, transcribe_audio_async
from modules.tts_service import initialize_tts, text_to_speech_async
from modules.weather_service import initialize_weather_service, get_weather_async
from modules.llm_service import initialize_llm, get_llm_response
from modules.intent_classifier import initialize_intent_classifier, detect_intent_async
from modules.reminder_utils import parse_reminder, parse_list_reminder_request
from modules.db_manager import initialize_db, save_reminder_async, get_reminders_for_date_async, reminder_check_loop
from modules.gui_utils import show_reminders_gui # type: ignore
from modules.retrain_utils import trigger_model_retraining_async, parse_retrain_request
from modules.contractions import normalize_text

# --- Modularized interaction logic ---

async def process_command(transcription: str):
    # Normalize contractions and pronunciation issues
    normalized_transcription = normalize_text(transcription)
    print(f"Processing command: {normalized_transcription}")
    intent = await detect_intent_async(normalized_transcription)
    response = ""
    if intent == "set_reminder":
        reminder = parse_reminder(normalized_transcription)
        if reminder:
            await save_reminder_async(reminder["task"], reminder["time"])
            response = f"Okay, I've set a reminder for '{reminder['task']}' at {reminder['time'].strftime('%I:%M %p on %A, %B %d')}"
        else:
            response = "I couldn't quite understand the reminder. Please try saying something like 'remind me to call John tomorrow at 2 pm'."
    elif intent == "list_reminders":
        target_date = parse_list_reminder_request(normalized_transcription or "")
        if target_date:
            reminders_found = await get_reminders_for_date_async(target_date)
            date_str = target_date.strftime('%A, %B %d, %Y')
            if reminders_found:
                response = f"Here are your reminders for {date_str}: "
                for r in reminders_found:
                    response += f"{r['task']} at {r['time'].strftime('%I:%M %p')}. "
            else:
                response = f"You have no reminders scheduled for {date_str}."
            # Show reminders in GUI and read out loud
            import threading
            threading.Thread(target=show_reminders_gui, args=(reminders_found, date_str), daemon=True).start() # type: ignore
        else:
            response = "I couldn't understand which date you want reminders for. Please specify a day like 'today', 'tomorrow', or a specific date."
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
        return
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
            r"(?:weather in|weather for|weather at|weather like in)\s+([A-Za-z\s,]+(?:\s+[A-Za-z]+)*)",
            normalized_transcription,
            re.IGNORECASE,
        )

        if location_match:
            extracted_location = location_match.group(1).strip()
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
                return  # Exit early as we need more information

        # Now, fetch weather based on determined location_name or use_current_location
        weather_data = None
        if use_current_location:
            print("Fetching weather for current location...")
            weather_data = await get_weather_async(
                None
            )  # Pass None to signify current location
            if weather_data:
                response = f"The current weather in {weather_data['city']} is {weather_data['description']} with a temperature of {weather_data['temp']:.1f} degrees Celsius."

            else:
                response = "Sorry, I couldn't determine your current location or fetch the weather for it. Please check your internet connection or try specifying a city."
        elif location_name:
            print(f"Fetching weather for {location_name}...")
            weather_data = await get_weather_async(location_name)
            if weather_data:
                response = f"The current weather in {weather_data['city']} is {weather_data['description']} with a temperature of {weather_data['temp']:.1f} degrees Celsius."
            else:
                response = f"Sorry, I couldn't fetch the weather for {location_name}. Please ensure the API key is set up and the location is valid."

        if not response:  # Fallback if no specific response was generated
            response = "I'm not sure which location you're asking about for the weather. Please specify, like 'weather in London' or 'weather in my area'."
    elif intent == "cancel_task":
        response = "Okay, cancelling that."
        # Need a more advanced cancel method, might need to manage/interrupt ongoing async tasks.
    elif intent == "calendar_query":
        response = "I'm not yet connected to your calendar, but I can set reminders."
    else:  # Default to general query if no other intent matched
        print("Sending to LLM for general query or unhandled/low-confidence intent...")
        response = await get_llm_response(
            input_text=normalized_transcription
        )  # LLM handles general queries
    print(f"Assistant: {response}")
    await text_to_speech_async(response)


async def handle_interaction():
    print(f"Assistant (speaking): {GREETING_MESSAGE}")
    await text_to_speech_async(GREETING_MESSAGE)
    audio_data = await record_audio_async()
    transcription = await transcribe_audio_async(audio_data)
    if not transcription or not transcription.strip():
        print("No speech detected after greeting.")
        await text_to_speech_async("I didn't catch that. If you need something, please call me again.")
        return
    print(f"User said: {transcription}")
    await process_command(transcription)


# Main logic
async def main():
    # Initialize all services
    initialize_db()
    initialize_stt()
    initialize_tts()
    initialize_weather_service()
    initialize_llm()
    initialize_intent_classifier()

    main_event_loop = asyncio.get_running_loop()
    asyncio.create_task(reminder_check_loop(text_to_speech_async)) # Run as an asyncio task

    def wakeword_callback():
        print("Wakeword detected! Initiating interaction.")
        future = asyncio.run_coroutine_threadsafe(handle_interaction(), main_event_loop)
        def future_done(f):
            try:
                f.result()
                print("Interaction task completed.")
            except Exception as e:
                print(f"Error in scheduled interaction task: {e}")
        future.add_done_callback(future_done)
    wakeword_thread = threading.Thread(
        target=run_wakeword, args=(wakeword_callback,), daemon=True
    )
    wakeword_thread.start()
    print("Voice assistant is running. Say the wakeword to interact.")
    try:
        while True:
            await asyncio.sleep(1.0)
    except KeyboardInterrupt:
        print("Main loop interrupted by user. Exiting.")
    finally:
        print("Main loop finished.")

if platform.system() == "Emscripten":
    asyncio.ensure_future(main())
else:
    if __name__ == "__main__":
        asyncio.run(main())
