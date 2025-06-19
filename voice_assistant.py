import asyncio
import platform
from typing import Optional
import warnings
import logging
import threading # Import the threading module


# --- Basic Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
# Define the logger at the module level so it's accessible throughout
logger = logging.getLogger(__name__)


from wakeword_detector import run_wakeword_async # type: ignore

from modules.audio_utils import record_audio_async
from modules.stt_service import initialize_stt, transcribe_audio_async
from modules.tts_service import initialize_tts, text_to_speech_async
from modules.weather_service import initialize_weather_service
from modules.llm_service import initialize_llm
from modules.intent_classifier import initialize_intent_classifier
from modules.db_manager import (
    initialize_db,
    reminder_check_loop,
)

# Import the new intent logic module
from modules.intent_logic import process_command, get_response, ShutdownSignal

# Import for validation and retraining logic
# from scripts.intent_validator import run_validation_and_retrain_async # This seems unused here
from modules.contractions import reload_normalization_data, NORMALIZATION_DATA_DIR # Import for watcher
from modules.file_watcher_service import start_normalization_data_watcher # Import watcher
# from modules.retrain_utils import parse_retrain_request  # This seems to be unused now

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

# Helper function to get a descriptive name for a task
def task_name(task: asyncio.Task) -> str:
    try:
        return task.get_name()  # Python 3.8+
    except AttributeError:
        return str(task)


async def handle_interaction():
    try:
        greeting = get_response("greeting") # Use imported get_response
        logging.info(f"Assistant (speaking): {greeting}") # Changed from print
        await text_to_speech_async(greeting)

        audio_data = None
        try:
            # Record audio with a 10-second timeout
            audio_data = await asyncio.wait_for(record_audio_async(), timeout=10.0)
        except asyncio.TimeoutError:
            logging.warning("Audio recording timed out.") # Changed from print
            await text_to_speech_async("Sorry, I couldn't capture audio in time.")
            return
        except Exception as rec_e:
            logging.error(f"Error during audio recording: {rec_e}") # Changed from print
            await text_to_speech_async(
                "Sorry, there was an issue with audio recording."
            )
            return

        if audio_data is None or not audio_data.any():  # Check if audio_data is valid
            logging.warning("No audio data captured or audio data is empty.") # Changed from print
            await text_to_speech_async(get_response("no_speech_detected")) # Use imported get_response
            return

        transcription = await transcribe_audio_async(audio_data)
        if not transcription or not transcription.strip():
            logging.info("No speech detected after greeting.") # Changed from print
            await text_to_speech_async(get_response("no_speech_detected")) # Use imported get_response
            return
        logging.info(f"User said: {transcription}") # Changed from print
        await process_command(transcription)
    except ShutdownSignal: # Catch the specific signal to propagate it
        raise # Re-raise to be caught by main_loop
    except Exception as e:
        logging.error(f"Exception in handle_interaction: {e}", exc_info=True) # Changed from print, added exc_info
        # Optionally, speak an error message here if it's not a shutdown
        await text_to_speech_async("An unexpected error occurred while handling your request.")


async def main_loop(loop: asyncio.AbstractEventLoop):
    # Keep track of tasks to cancel them on exit
    background_tasks = []
    reminder_task = loop.create_task(reminder_check_loop(text_to_speech_async))
    background_tasks.append(reminder_task)

    # Start the file watcher for normalization data in a separate thread
    # This thread will run in the background and call reload_normalization_data when files change.
    # We need to ensure this thread is managed correctly on shutdown.
    normalization_files_to_watch = ["contractions_map.json", "common_misspellings_map.json", "custom_dictionary.txt"]
    watcher_thread = threading.Thread(
        target=start_normalization_data_watcher,
        args=(reload_normalization_data, NORMALIZATION_DATA_DIR, normalization_files_to_watch),
        daemon=True # Daemonize so it exits when the main program exits
    )
    watcher_thread.start()

    current_wakeword_task: Optional[asyncio.Task] = None

    try:
        while True:
            logging.info("Waiting for wake word...") # Changed from print
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
                logging.info("Main loop's wait for wake_event cancelled.") # Changed from print
                raise

            if current_wakeword_task:
                current_wakeword_task.cancel()
                try:
                    await current_wakeword_task
                except asyncio.CancelledError:
                    pass

            try:
                await handle_interaction()
            except ShutdownSignal:
                logging.info("Shutdown signal received in main_loop. Exiting loop.") # Changed from print
                break # Exit the while True loop to start graceful shutdown
            except Exception as e: # Catch other errors from handle_interaction
                logging.error(f"Error in interaction cycle: {e}", exc_info=True) # Changed from print, added exc_info
    except asyncio.CancelledError:
        logging.info("Main loop was cancelled.") # Changed from print
    finally:
        logging.info("Main loop ending. Cancelling background tasks...") # Changed from print
        if current_wakeword_task and not current_wakeword_task.done():
            current_wakeword_task.cancel()
            try:
                await current_wakeword_task
            except asyncio.CancelledError:
                logging.info("Wakeword task cancelled during main loop cleanup.") # Changed from print

        for task in background_tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    logging.info(f"Background task {task_name(task)} was cancelled.") # Changed from print
                except Exception as e:
                    logging.error(
                        f"Error during cancellation of background task {task_name(task)}: {e}", exc_info=True
                    )
        # The watcher_thread is a daemon, so it should exit automatically.
        # If explicit cleanup for the observer was needed outside its own try/finally,
        # you'd need a way to signal it to stop (e.g., an event) and then join it.
        # However, the observer's own try/finally with observer.stop() should handle it.
        logging.info("File watcher thread (daemon) will exit with main program.") # Use logging directly
        logging.info("All background tasks in main_loop processed for cancellation.") # Changed from print


def run_assistant():
    # --- Initialization ---
    logging.info("Initializing services...")
    # For Windows, set the policy to allow more threads if needed
    if platform.system() == "Windows":
        import nest_asyncio

        nest_asyncio.apply()

    loop = asyncio.get_event_loop()  # Get the event loop
    logging.info("Initializing services...") # Moved here from print
    initialize_stt()
    initialize_tts()
    initialize_weather_service()
    initialize_llm()
    initialize_intent_classifier()
    initialize_db()

    logging.info("Services initialized. You can now interact with the assistant.") # Changed from print

    main_task: Optional[asyncio.Task] = None
    try:
        main_task = loop.create_task(main_loop(loop))
        loop.run_until_complete(main_task)
    except KeyboardInterrupt:
        logging.info("\nKeyboardInterrupt received. Shutting down gracefully...") # Changed from print
        if main_task and not main_task.done():
            main_task.cancel()
            try:
                # Give main_task a chance to clean up
                loop.run_until_complete(main_task)
            except asyncio.CancelledError:
                logging.info("Main task successfully cancelled.") # Changed from print
            except RuntimeError as e:  # Handle cases where loop might be closing
                logging.error(f"Runtime error during main_task cleanup: {e}", exc_info=True) # Changed from print
            except Exception as e:
                logging.error(f"Exception during main_task cleanup: {e}", exc_info=True) # Changed from print
    except Exception as e:  # Catch other unexpected errors from main_loop
        logging.critical(f"Unexpected error in run_assistant: {e}", exc_info=True) # Changed from print
        if (
            main_task and not main_task.done()
        ):  # Attempt to cancel main_task if it's still running
            main_task.cancel()
            if loop.is_running():
                loop.run_until_complete(main_task)  # Allow it to process cancellation
    finally:
        logging.info("Performing final cleanup of any remaining tasks...") # Changed from print
        remaining_tasks = [
            t
            for t in asyncio.all_tasks(loop=loop)
            if t is not asyncio.current_task(loop=loop) and not t.done()
        ]
        if remaining_tasks:
            logging.info(f"Cancelling {len(remaining_tasks)} remaining tasks...") # Changed from print
            for task in remaining_tasks:
                task.cancel()
            if loop.is_running():
                loop.run_until_complete(
                    asyncio.gather(*remaining_tasks, return_exceptions=True)
                )
        logging.info("Closing event loop.") # Changed from print
        if not loop.is_closed():
            loop.close()
        logging.info("Assistant shut down.") # Changed from print


if __name__ == "__main__":
    run_assistant()
