"""
Module for setup and selection of speech-to-text (STT) models: WhisperX and Whisper.
Provides user with descriptions, pros, and cons for each model.
"""
import os

def describe_stt_models():
    return {
        "whisperx": {
            "name": "WhisperX",
            "description": "Fast, accurate, and supports word-level timestamps and speaker diarization. Built on top of OpenAI Whisper with extra features.",
            "pros": [
                "Faster than standard Whisper (with batch and alignment)",
                "Supports word-level timestamps",
                "Speaker diarization (who spoke when)",
                "Good for long audios and advanced use cases"
            ],
            "cons": [
                "Requires more dependencies (whisperx, sounddevice, scipy)",
                "Slightly more complex setup",
                "May use more memory"
            ]
        },
        "whisper": {
            "name": "Whisper (OpenAI)",
            "description": "The original OpenAI Whisper model. Simple, robust, and easy to use for most speech-to-text tasks.",
            "pros": [
                "Simple setup",
                "Good accuracy",
                "Lower resource usage than WhisperX for short audios"
            ],
            "cons": [
                "No word-level timestamps",
                "No built-in speaker diarization",
                "Slower for long audios"
            ]
        }
    }


def prompt_stt_model_choice():
    models = describe_stt_models()
    print("\nAvailable Speech-to-Text Models:")
    for key, info in models.items():
        print(f"\n[{key}] {info['name']}")
        print(f"  {info['description']}")
        print("  Pros:")
        for pro in info['pros']:
            print(f"    + {pro}")
        print("  Cons:")
        for con in info['cons']:
            print(f"    - {con}")
    while True:
        choice = input("\nEnter the model you want to use (whisperx/whisper): ").strip().lower()
        if choice in models:
            return choice
        print("Invalid choice. Please enter 'whisperx' or 'whisper'.")
