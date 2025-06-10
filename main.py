# voice_assistant/main.py
import time
import os
import sys
import importlib.util
from core.engine import VoiceCore
from core.tts import tts_engine, speak  # Import speak for unrecognized commands


class Assistant:
    def __init__(self):
        """Initializes the Assistant, loads intents, and starts the core."""
        self.intents = {}
        self._load_modules()

        self.core = VoiceCore(
            wake_word_model_path="./hey_jimmy.onnx",
            on_wake_word=self.handle_wake_word,
            on_command=self.handle_command,
        )

    def _load_modules(self):
        """Dynamically loads all intent modules from the 'modules' directory."""
        print("Loading intent modules...")
        modules_dir = os.path.join(os.path.dirname(__file__), "modules")
        for filename in os.listdir(modules_dir):
            if filename.endswith(".py") and not filename.startswith("__"):
                module_name = filename[:-3]
                try:
                    spec = importlib.util.spec_from_file_location(
                        module_name, os.path.join(modules_dir, filename)
                    )
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)

                    if hasattr(module, "register_intents"):
                        intents_from_module = module.register_intents()
                        self.intents.update(intents_from_module)
                        print(f"  - Loaded intents from '{filename}'")
                except Exception as e:
                    print(f"  - FAILED to load '{filename}': {e}")
        print(f"Total intents loaded: {len(self.intents)}")

    def handle_wake_word(self):
        """Called by the core when 'Hey Jimmy' is detected."""
        print("\nAssistant: Wake word acknowledged. Listening for command...")
        speak("Yes?")

    def handle_command(self, command: str):
        """
        Called by the core with the transcribed command.
        Finds and executes the appropriate action from the loaded intents.
        """
        print(f"Assistant: Received command -> '{command}'")
        command = command.lower().strip()

        for intent_phrase, action in self.intents.items():
            if intent_phrase in command:
                print(f"Assistant: Matched intent '{intent_phrase}'. Executing action.")
                action()
                return

        print("Assistant: Command not recognized.")
        speak("Sorry, I don't know how to do that yet.")

    def run(self):
        """Starts the core engine and keeps the application alive."""
        self.core.start()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down assistant...")
            speak("Goodbye!")
            # Gracefully stop the core engines
            tts_engine.stop()
            self.core.stop()


if __name__ == "__main__":
    assistant = Assistant()
    assistant.run()
