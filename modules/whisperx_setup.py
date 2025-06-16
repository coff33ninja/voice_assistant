def setup_whisperx():
    try:
        import whisperx
        try:
            import sounddevice as sd
            import scipy.io.wavfile as wav
        except ImportError:
            print("sounddevice and scipy are required. Please install them with: pip install sounddevice scipy")
            return
        import tempfile
        import os
        print("WhisperX setup complete. Models will download on first use.")
        # Prompt user to say something
        duration = 5  # seconds
        fs = 16000
        print("Please say something after the beep...")
        sd.sleep(500)
        print("Beep!")
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
        sd.wait()
        # Save to temp wav file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmpfile:
            wav.write(tmpfile.name, fs, recording)
            tmp_wav_path = tmpfile.name
        # Transcribe with whisperx
        model = whisperx.load_model("base", device="cpu", compute_type="int8")
        result = model.transcribe(tmp_wav_path)
        # Print the transcription result (print the whole result for clarity)
        print("Transcription result:", result)
        os.remove(tmp_wav_path)
    except Exception as e:
        print(f"WhisperX setup or test failed: {e}")