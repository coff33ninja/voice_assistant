import asyncio
import numpy as np
import whisperx
from .config import ASR_DEVICE, ALIGN_LANGUAGE_CODE, STT_MODEL_NAME, STT_COMPUTE_TYPE, STT_BATCH_SIZE

stt_model_global = None
align_model_global = None
align_metadata_global = None

def initialize_stt():
    global stt_model_global, align_model_global, align_metadata_global
    print("Initializing STT service...")
    stt_model_global = whisperx.load_model(
        STT_MODEL_NAME, device=ASR_DEVICE, compute_type=STT_COMPUTE_TYPE
    )
    try:
        align_model_global, align_metadata_global = whisperx.load_align_model(
            language_code=ALIGN_LANGUAGE_CODE, device=ASR_DEVICE
        )
        print(f"Alignment model for '{ALIGN_LANGUAGE_CODE}' loaded.")
    except Exception as e:
        print(f"Warning: Failed to load alignment model for '{ALIGN_LANGUAGE_CODE}': {e}. Alignment will be skipped.")
        align_model_global, align_metadata_global = None, None
    print("STT service initialized.")

async def transcribe_audio_async(audio_data_np_int16: np.ndarray) -> str:
    if stt_model_global is None:
        raise RuntimeError("STT service not initialized. Call initialize_stt() first.")

    audio_float32 = audio_data_np_int16.astype(np.float32) / 32768.0
    transcribe_result = await asyncio.to_thread(stt_model_global.transcribe, audio_float32, batch_size=STT_BATCH_SIZE)

    if not transcribe_result or not transcribe_result.get("segments"):
        return ""

    if align_model_global and align_metadata_global:
        current_lang_code = transcribe_result["language"]
        if current_lang_code == ALIGN_LANGUAGE_CODE:
            await asyncio.to_thread(
                whisperx.align,
                transcribe_result["segments"],
                align_model_global,
                align_metadata_global,
                audio_float32,
                ASR_DEVICE,
                current_lang_code
            )
        else:
            print(
                f"Warning (Async STT): Transcription language '{current_lang_code}' does not match "
                f"loaded alignment model language '{ALIGN_LANGUAGE_CODE}'. Skipping alignment."
            )
    else:
        print("Warning: Alignment model or metadata not available. Skipping alignment in async STT.")

    texts = [segment.get("text", "").strip() for segment in transcribe_result["segments"]]
    transcription = " ".join(filter(None, texts))
    return transcription

def transcribe_audio(audio_data_np_int16: np.ndarray) -> str:
    if stt_model_global is None:
        raise RuntimeError("STT service not initialized. Call initialize_stt() first.")
    audio_float32 = audio_data_np_int16.astype(np.float32) / 32768.0
    result = stt_model_global.transcribe(audio_float32, batch_size=STT_BATCH_SIZE)
    if not result or not result.get("segments"):
        return ""
    if align_model_global and align_metadata_global:
        current_lang_code = result["language"]
        if current_lang_code == ALIGN_LANGUAGE_CODE:
            whisperx.align(result["segments"], align_model_global, align_metadata_global, audio_float32, ASR_DEVICE, current_lang_code)
        else:
            print(f"Warning: Transcription language '{current_lang_code}' differs from alignment model '{ALIGN_LANGUAGE_CODE}'. Skipping.")
    else:
        print("Warning: Alignment model/metadata not loaded. Skipping alignment.")
    texts = [segment.get("text", "").strip() for segment in result["segments"]]
    return " ".join(filter(None, texts))