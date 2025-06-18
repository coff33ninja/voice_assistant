import os
import asyncio
import sounddevice as sd
from precise_runner import PreciseEngine, PreciseRunner
import logging
import pvporcupine
from modules.config import (
    PRECISE_ENGINE_EXECUTABLE,
    PRECISE_MODEL_HEY_MIKA,
    PICOVOICE_KEY_FILE_PATH, # Renamed from PICOVOICE_KEY_FILE for clarity
    PICOVOICE_MODEL_HEY_MIKA,
    get_picovoice_key as get_picovoice_key_from_config, # Import the config function
)

logger = logging.getLogger(__name__)

def get_porcupine_key():
    # Prioritize environment variable, then file, consistent with config.py
    env_key = get_picovoice_key_from_config()
    if env_key:
        return env_key
    if os.path.exists(PICOVOICE_KEY_FILE_PATH): # Use the correct imported path
        with open(PICOVOICE_KEY_FILE_PATH, "r") as f:
            return f.read().strip()
    return None # Return None if key file doesn't exist

async def detect_wakeword_precise(callback):
    try:
        if not os.path.exists(PRECISE_MODEL_HEY_MIKA):
            raise FileNotFoundError("Precise wakeword model not found. Train a model first.")
        engine = PreciseEngine(PRECISE_ENGINE_EXECUTABLE, PRECISE_MODEL_HEY_MIKA)
        runner = PreciseRunner(engine, on_activation=callback, sensitivity=0.5)
        await asyncio.to_thread(runner.start) # Run blocking start in a thread
        while True:
            await asyncio.sleep(1.0)
    except Exception as e:
        logger.error(f"Precise wakeword error: {e}. Ensure model is trained and placed at {PRECISE_MODEL_HEY_MIKA}.")
        # runner.stop() might be needed here if it was started.
        # PreciseRunner's start() is blocking if not handled, or runs its own thread. # type: ignore
        # If runner.start() itself fails, this is fine. If it starts then an error occurs, cleanup might be needed.
        # Ensure proper cleanup of PreciseRunner resources.
        finally:
            if 'runner' in locals() and runner is not None:
                logger.info("Stopping PreciseRunner to release resources...")
                runner.stop()

async def detect_wakeword_porcupine(callback, access_key):
    porcupine = None
    stream = None
    try:
        logger.info("Initializing Porcupine...")
        porcupine = await asyncio.to_thread(pvporcupine.create, access_key=access_key, keyword_paths=[PICOVOICE_MODEL_HEY_MIKA])
        logger.info("Porcupine initialized.")

        sample_rate = porcupine.sample_rate
        frame_length = porcupine.frame_length

        logger.info("Initializing audio stream...")
        stream = sd.InputStream(
            samplerate=sample_rate, channels=1, dtype='int16', blocksize=frame_length
        )
        await asyncio.to_thread(stream.start)
        logger.info("Audio stream started. Listening for wakeword...")

        while True:
            audio_data, overflowed = await asyncio.to_thread(stream.read, frame_length)
            if overflowed:
                logger.warning("Audio buffer overflowed during Porcupine detection")

            # Porcupine expects a list of int16 PCM samples
            audio_pcm_list = audio_data.flatten().tolist()

            keyword_index = await asyncio.to_thread(porcupine.process, audio_pcm_list)

            if keyword_index >= 0:
                logger.info("Wakeword detected by Porcupine!")
                callback() # Synchronous callback

            await asyncio.sleep(0.01) # Yield control briefly

    except pvporcupine.PorcupineError as pe:
        logger.error(f"Porcupine engine error: {pe}.")
        raise
    except sd.PortAudioError as pae:
        logger.error(f"Sounddevice/PortAudio error: {pae}. Check audio devices.")
        raise
    except Exception as e: # Catch any other unexpected errors
        logger.error(f"Unexpected error in Porcupine wakeword detection: {e}.", exc_info=True)
        raise  # Re-raise the exception to be handled by the caller
    finally:
        if stream is not None:
            logger.info("Stopping and closing Porcupine audio stream...")
            await asyncio.to_thread(stream.stop)
            await asyncio.to_thread(stream.close)
        if porcupine is not None:
            logger.info("Deleting Porcupine instance...")
            await asyncio.to_thread(porcupine.delete)

async def detect_wakeword(callback):
    access_key = get_porcupine_key()
    porcupine_model_available = access_key and os.path.exists(PICOVOICE_MODEL_HEY_MIKA)

    if porcupine_model_available:
        logger.info(f"Picovoice key found and Porcupine model exists at {PICOVOICE_MODEL_HEY_MIKA}.")
        logger.info("Attempting to use Porcupine for wakeword detection...")
        try:
            await detect_wakeword_porcupine(callback, access_key)
            # If detect_wakeword_porcupine runs indefinitely and successfully, execution stops here.
            return
        except Exception as e_porcupine:
            logger.error(f"Porcupine engine failed to start or encountered an error: {e_porcupine}.")
            logger.info("Attempting to fall back to Precise engine.")
            # Fall-through to Precise logic
    else:
        if not access_key:
            logger.info("Picovoice access key not found or not provided.")
        elif not os.path.exists(PICOVOICE_MODEL_HEY_MIKA): # Check if model is the missing part
            logger.info(f"Porcupine model not found at {PICOVOICE_MODEL_HEY_MIKA}.")
        logger.info("Proceeding to use Precise engine as primary or fallback.")

    # Attempt to use Precise engine if Porcupine was not used or failed.
    # The detect_wakeword_precise function will raise FileNotFoundError if its model is missing,
    # which will then be caught and printed by its own exception handler.
    logger.info(f"Attempting to use Precise engine with model: {PRECISE_MODEL_HEY_MIKA}")
    try:
        await detect_wakeword_precise(callback)
    except Exception as e_precise: # Catch potential errors from precise, though it handles its own printing
        logger.error(f"Failed to run Precise engine: {e_precise}") # This provides context if precise itself fails beyond model not found

async def run_wakeword_async(callback): # Renamed to indicate it's an async function
    """Asynchronously runs the wakeword detection."""
    await detect_wakeword(callback)

if __name__ == "__main__":
    def test_callback():
        logger.info("Wakeword callback triggered (test).")
    asyncio.run(run_wakeword_async(test_callback)) # Test with asyncio.run if run directly
