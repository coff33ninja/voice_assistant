import unittest
from core import tts

class TestTTSEngine(unittest.TestCase):
    def setUp(self):
        self.engine = tts.TTSEngine()

    def test_speak_and_interrupt(self):
        # Should not raise
        self.engine.speak("Hello, world!")
        self.engine.interrupt()
        self.assertTrue(True)  # If no exception, pass

    def test_set_voice_invalid(self):
        result = self.engine.set_voice("nonexistent_voice_id")
        self.assertFalse(result)

    def test_get_available_voices(self):
        voices = self.engine.get_available_voices()
        self.assertIsInstance(voices, list)
        self.assertGreater(len(voices), 0)
        self.assertIn('id', voices[0])

    def test_stop(self):
        # Should stop gracefully
        self.engine.stop()
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
