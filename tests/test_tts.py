import unittest
from core import tts

class TestTTSEngine(unittest.TestCase):
    def setUp(self):
        """
        Initializes a new TTSEngine instance before each test method is run.
        """
        self.engine = tts.TTSEngine()

    def test_speak_and_interrupt(self):
        # Should not raise
        """
        Tests that calling speak followed by interrupt does not raise exceptions.
        """
        self.engine.speak("Hello, world!")
        self.engine.interrupt()
        self.assertTrue(True)  # If no exception, pass

    def test_set_voice_invalid(self):
        """
        Tests that setting an invalid voice ID returns False.
        """
        result = self.engine.set_voice("nonexistent_voice_id")
        self.assertFalse(result)

    def test_get_available_voices(self):
        """
        Tests that get_available_voices returns a non-empty list of voice dictionaries containing an 'id' key.
        """
        voices = self.engine.get_available_voices()
        self.assertIsInstance(voices, list)
        self.assertGreater(len(voices), 0)
        self.assertIn('id', voices[0])

    def test_stop(self):
        # Should stop gracefully
        """
        Tests that the TTSEngine's stop method completes without raising exceptions.
        """
        self.engine.stop()
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
