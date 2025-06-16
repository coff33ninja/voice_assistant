import os
import asyncio
import sounddevice as sd
from precise_runner import PreciseEngine, PreciseRunner
import pvporcupine

# Determine the absolute path to the 'models' directory relative to this script
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(_SCRIPT_DIR, "models")

PRECISE_ENGINE = os.path.join(BASE_DIR, "precise-engine/precise-engine")
PRECISE_MODEL = os.path.join(BASE_DIR, "hey_mika.pb") # Renamed wakeword model
PICOVOICE_KEY_FILE = os.path.join(BASE_DIR, "picovoice_key.txt")
PICOVOICE_MODEL = os.path.join(BASE_DIR, "hey_mika.ppn") # Renamed wakeword model

def get_porcupine_key():
    if os.path.exists(PICOVOICE_KEY_FILE):
        with open(PICOVOICE_KEY_FILE, "r") as f:
            return f.read().strip()
    return None

async def detect_wakeword_precise(callback):
    try:
        if not os.path.exists(PRECISE_MODEL):
            raise FileNotFoundError("Precise wakeword model not found. Train a model first.")
        engine = PreciseEngine(PRECISE_ENGINE, PRECISE_MODEL)
        runner = PreciseRunner(engine, on_activation=callback, sensitivity=0.5)
        await asyncio.to_thread(runner.start) # Run blocking start in a thread
        while True:
            await asyncio.sleep(1.0)
    except Exception as e:
        print(f"Precise wakeword error: {e}. Ensure model is trained and placed at {PRECISE_MODEL}.")
        # runner.stop() might be needed here if it was started.
        # PreciseRunner's start() is blocking if not handled, or runs its own thread.
        # If runner.start() itself fails, this is fine. If it starts then an error occurs, cleanup might be needed.
        # For now, assume PreciseRunner handles its own cleanup on error or relies on daemon thread.

async def detect_wakeword_porcupine(callback, access_key):
    porcupine = None
    stream = None
    try:
        print("Initializing Porcupine...")
        porcupine = await asyncio.to_thread(pvporcupine.create, access_key=access_key, keyword_paths=[PICOVOICE_MODEL])
        print("Porcupine initialized.")

        sample_rate = porcupine.sample_rate
        frame_length = porcupine.frame_length

        print("Initializing audio stream...")
        stream = sd.InputStream(
            samplerate=sample_rate, channels=1, dtype='int16', blocksize=frame_length
        )
        await asyncio.to_thread(stream.start)
        print("Audio stream started. Listening for wakeword...")

        while True:
            audio_data, overflowed = await asyncio.to_thread(stream.read, frame_length)
            if overflowed:
                print("Warning: audio buffer overflowed")
            
            # Porcupine expects a list of int16 PCM samples
            audio_pcm_list = audio_data.flatten().tolist()
            
            keyword_index = await asyncio.to_thread(porcupine.process, audio_pcm_list)
            
            if keyword_index >= 0:
                print("Wakeword detected by Porcupine!")
                callback() # Synchronous callback
            
            await asyncio.sleep(0.01) # Yield control briefly

    except pvporcupine.PorcupineError as pe:
        print(f"Porcupine engine error: {pe}.")
        raise
    except sd.PortAudioError as pae:
        print(f"Sounddevice/PortAudio error: {pae}. Check audio devices.")
        raise
    except Exception as e: # Catch any other unexpected errors
        print(f"Unexpected error in Porcupine wakeword detection: {e}.")
        raise  # Re-raise the exception to be handled by the caller
    finally:
        if stream is not None:
            print("Stopping and closing audio stream...")
            await asyncio.to_thread(stream.stop)
            await asyncio.to_thread(stream.close)
        if porcupine is not None:
            print("Deleting Porcupine instance...")
            await asyncio.to_thread(porcupine.delete)

async def detect_wakeword(callback):
    access_key = get_porcupine_key()
    porcupine_model_available = access_key and os.path.exists(PICOVOICE_MODEL)

    if porcupine_model_available:
        print(f"Picovoice key found and Porcupine model exists at {PICOVOICE_MODEL}.")
        print("Attempting to use Porcupine for wakeword detection...")
        try:
            await detect_wakeword_porcupine(callback, access_key)
            # If detect_wakeword_porcupine runs indefinitely and successfully, execution stops here.
            return
        except Exception as e_porcupine:
            print(f"Porcupine engine failed to start or encountered an error: {e_porcupine}.")
            print("Attempting to fall back to Precise engine.")
            # Fall-through to Precise logic
    else:
        if not access_key:
            print("Picovoice access key not found or not provided.")
        elif not os.path.exists(PICOVOICE_MODEL): # Check if model is the missing part
            print(f"Porcupine model not found at {PICOVOICE_MODEL}.")
        print("Proceeding to use Precise engine as primary or fallback.")

    # Attempt to use Precise engine if Porcupine was not used or failed.
    # The detect_wakeword_precise function will raise FileNotFoundError if its model is missing,
    # which will then be caught and printed by its own exception handler.
    print(f"Attempting to use Precise engine with model: {PRECISE_MODEL}")
    try:
        await detect_wakeword_precise(callback)
    except Exception as e_precise: # Catch potential errors from precise, though it handles its own printing
        print(f"Failed to run Precise engine: {e_precise}") # This provides context if precise itself fails beyond model not found

def run_wakeword(callback):
    asyncio.run(detect_wakeword(callback))

if __name__ == "__main__":
    def test_callback():
        print("Wakeword callback triggered.")
    run_wakeword(test_callback)