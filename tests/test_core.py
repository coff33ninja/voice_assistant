# voice_assistant/tests/test_core.py
import unittest
import os
import whisper
import subprocess

# This is a bit of a trick to import from the parent directory
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.engine import openwakeword


class TestCoreComponents(unittest.TestCase):

    def test_wakeword_model_exists(self):
        """Test 1: Check if the wake word model file exists."""
        print("\nRunning Test 1: Wake Word Model Existence")
        model_path = "./hey_jimmy.onnx"
        self.assertTrue(
            os.path.exists(model_path), f"Wake word model not found at {model_path}"
        )
        print("✅ Test 1 Passed: Wake word model found.")

    def test_wakeword_model_loads(self):
        """Test 2: Check if openWakeWord can load the model."""
        print("\nRunning Test 2: Wake Word Model Loading")
        try:
            oww_model = openwakeword.Model(wakeword_models=["./hey_jimmy.onnx"])
            self.assertIsNotNone(oww_model, "Model loading returned None.")
            print("✅ Test 2 Passed: openWakeWord model loaded successfully.")
        except Exception as e:
            self.fail(f"openWakeWord model failed to load. Error: {e}")

    def test_whisper_model_loads(self):
        """Test 3: Check if Whisper can load its model."""
        print("\nRunning Test 3: Whisper Model Loading")
        try:
            whisper_model = whisper.load_model("base.en")
            self.assertIsNotNone(whisper_model, "Whisper model loading returned None.")
            print("✅ Test 3 Passed: OpenAI Whisper model loaded successfully.")
        except Exception as e:
            self.fail(f"Whisper model failed to load. Error: {e}")


class TestGeneralFunctions(unittest.TestCase):
    def test_run_self_test(self):
        """Test the run_self_test function."""
        try:
            subprocess.run(
                [sys.executable, "run_tests.py"], capture_output=True, text=True, check=True
            )
        except FileNotFoundError:
            self.fail("'run_tests.py' not found.")


if __name__ == "__main__":
    unittest.main()
