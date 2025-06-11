# voice_assistant/tests/test_core.py
import unittest
import os
import whisper
import subprocess
import sys

# This is a bit of a trick to import from the parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.engine import openwakeword, VoiceCore


class TestCoreComponents(unittest.TestCase):
    def test_wakeword_model_exists(self):
        """
        Checks that the wake word model file exists at the expected path.
        
        Fails the test if the model file is missing.
        """
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
        """
        Tests that the OpenAI Whisper speech recognition model loads successfully.
        
        Fails if the model cannot be loaded or if an exception is raised during loading.
        """
        print("\nRunning Test 3: Whisper Model Loading")
        try:
            whisper_model = whisper.load_model("base.en")
            self.assertIsNotNone(whisper_model, "Whisper model loading returned None.")
            print("✅ Test 3 Passed: OpenAI Whisper model loaded successfully.")
        except Exception as e:
            self.fail(f"Whisper model failed to load. Error: {e}")

    def test_register_intents_exists(self):
        """
        Checks that the VoiceCore class has a callable load_intents method.
        
        Asserts that the VoiceCore engine includes an attribute named 'load_intents' and that it is callable, ensuring intent registration functionality is present.
        """
        print("\nRunning Test 4: Intent Registration Existence")
        self.assertTrue(hasattr(VoiceCore, "load_intents"), "Engine has no attribute 'load_intents'")
        self.assertTrue(callable(VoiceCore.load_intents), "'load_intents' is not callable")
        print("✅ Test 4 Passed: Intent registration functions exist.")

    def test_load_intents_smoke(self):
        """
        Smoke tests the VoiceCore.load_intents() method to ensure it executes without raising exceptions.
        """
        print("\nRunning Smoke Test 1: Intent Loading")
        try:
            VoiceCore.load_intents()
            print("✅ Smoke Test 1 Passed: Intents loaded successfully.")
        except Exception:
            self.fail("load_intents() raised an exception")


class TestGeneralFunctions(unittest.TestCase):
    def test_run_self_test(self):
        """
        Runs the external 'run_tests.py' script and fails if the script is not found.
        
        This test ensures that the self-test script executes successfully using the current Python interpreter.
        """
        try:
            subprocess.run(
                [sys.executable, "run_tests.py"], capture_output=True, text=True, check=True
            )
        except FileNotFoundError:
            self.fail("'run_tests.py' not found.")


if __name__ == "__main__":
    unittest.main()
