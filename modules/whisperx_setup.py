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
        import sys # for sys.executable
        import subprocess # for subprocess.run
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

        # Attempt to apply PyTorch Lightning checkpoint upgrade
        try:
            print("\nAttempting to apply PyTorch Lightning checkpoint upgrade for WhisperX model...")
            whisperx_module_path = os.path.dirname(whisperx.__file__)
            checkpoint_file_path = os.path.join(whisperx_module_path, "assets", "pytorch_model.bin")

            if os.path.exists(checkpoint_file_path):
                upgrade_command = [
                    sys.executable,
                    "-m",
                    "pytorch_lightning.utilities.upgrade_checkpoint",
                    checkpoint_file_path,
                ]
                print(f"Running: {' '.join(upgrade_command)}")
                process = subprocess.run(upgrade_command, capture_output=True, text=True, check=False)
                if process.returncode == 0:
                    print("PyTorch Lightning checkpoint upgrade command executed successfully (if an upgrade was needed).")
                    if process.stdout.strip():
                        print(f"Upgrade stdout:\n{process.stdout.strip()}")
                    # PTL might output to stderr even on success for info like "No upgrade needed"
                    if process.stderr.strip():
                        print(f"Upgrade stderr:\n{process.stderr.strip()}")
                else:
                    print(f"PyTorch Lightning checkpoint upgrade command may have failed or was not applicable. Return code: {process.returncode}")
                    if process.stdout.strip():
                        print(f"Stdout:\n{process.stdout.strip()}")
                    if process.stderr.strip():
                        print(f"Stderr:\n{process.stderr.strip()}")
                    print("This is usually not critical. The model should still work.")
            else:
                print(f"WhisperX checkpoint file not found at {checkpoint_file_path}. Skipping upgrade attempt.")
        except ImportError:
            print("pytorch_lightning module not found. Skipping checkpoint upgrade attempt.")
        except Exception as upgrade_e:
            print(f"An error occurred during PyTorch Lightning checkpoint upgrade attempt: {upgrade_e}")

    except Exception as e:
        print(f"WhisperX setup or test failed: {e}")