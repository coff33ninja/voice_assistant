# voice_assistant/main.py
import time
import os
import sys
import importlib.util
import subprocess
from core.engine import VoiceCore
from core.tts import tts_engine, speak
from core.user_config import load_config


class Assistant:
    def __init__(self):
        """Initializes the Assistant, loads intents, and starts the core."""
        # --- First Run Setup Check ---
        self.config = load_config()
        if not self.config.get("first_run_complete", False):
            print("INFO: First run setup not complete or config missing. Running setup script...")
            try:
                script_dir = os.path.dirname(os.path.abspath(__file__))
                setup_script_path = os.path.join(script_dir, "first_run_setup.py")
                if not os.path.exists(setup_script_path):
                    setup_script_path = os.path.join(os.path.dirname(script_dir), "first_run_setup.py")
                if not os.path.exists(setup_script_path):
                    print(f"CRITICAL: first_run_setup.py not found at expected locations (tried {os.path.join(script_dir, 'first_run_setup.py')} and {os.path.join(os.path.dirname(script_dir), 'first_run_setup.py')}).")
                    sys.exit(1)
                completed_process = subprocess.run([sys.executable, setup_script_path], check=False)
                if completed_process.returncode != 0:
                    print("ERROR: First run setup script exited with an error. Please check its output.")
                else:
                    print("First run setup script completed. Please restart the application.")
            except Exception as e_setup:
                print(f"ERROR: Failed to run first_run_setup.py: {e_setup}")
            sys.exit(0) # Exit after attempting setup, user needs to restart

        # --- Apply chosen TTS voice ---
        chosen_tts_voice_id = self.config.get("chosen_tts_voice_id")
        if chosen_tts_voice_id:
            print(f"INFO: Applying chosen TTS voice ID: {chosen_tts_voice_id}")
            # tts_engine is globally imported from core.tts
            if 'tts_engine' in globals() and hasattr(tts_engine, 'set_voice'):
                if not tts_engine.set_voice(chosen_tts_voice_id):
                    print(f"WARN: Failed to set chosen TTS voice ID '{chosen_tts_voice_id}'. Default will be used.")
            else:
                print("WARN: tts_engine not available or does not have set_voice method.")
        else:
            print("INFO: No chosen TTS voice ID found in config. Using default TTS voice.")

        self.picovoice_access_key = os.environ.get("PICOVOICE_ACCESS_KEY")
        if not self.picovoice_access_key:
            # This info is useful if user later tries to switch to Picovoice via config editing
            # or if first_run_setup had an issue but config was partially saved.
            print("INFO: PICOVOICE_ACCESS_KEY environment variable not set. This is required if using the Picovoice engine.")

        self.intents = {}
        self._load_modules()

        # --- Initialize Voice Core based on User Configuration ---
        engine_to_use = self.config.get("chosen_wake_word_engine")
        model_path = self.config.get("chosen_wake_word_model_path")
        access_key_env_set = self.config.get("picovoice_access_key_is_set_env", False)
        # self.picovoice_access_key is already loaded from env
        effective_picovoice_access_key = self.picovoice_access_key

        vc_args = {}
        vc_args['on_wake_word'] = self.handle_wake_word
        vc_args['on_command'] = self.handle_command
        # Default whisper_model_name and silence_threshold from VoiceCore's __init__ will be used

        if engine_to_use == "picovoice" and model_path and effective_picovoice_access_key and access_key_env_set:
            print(f"INFO: Using Picovoice engine with model: {model_path}")
            vc_args['engine_type'] = "picovoice"
            vc_args['picovoice_access_key'] = effective_picovoice_access_key
            vc_args['picovoice_keyword_paths'] = [model_path] # Porcupine expects a list
        else:
            if engine_to_use == "picovoice": # Specifically chosen but conditions not met
                print("WARN: Picovoice was configured but not all conditions are met (e.g., model path missing or PICOVOICE_ACCESS_KEY not found/confirmed during setup). Defaulting to OpenWakeWord.")
            elif engine_to_use is None: # No configuration set yet
                print("INFO: No wake word engine configured yet. Defaulting to OpenWakeWord.")
            else: # Configured to something else (e.g. openwakeword explicitly)
                print("INFO: Using configured OpenWakeWord engine.")

            # Fallback to OpenWakeWord
            # WAKE_WORD_MODEL_PATH env var is the primary source for OpenWakeWord if not set by user config.
            # The previous VoiceCore init used "./hey_jimmy.onnx" as a hardcoded default if env var wasn't set.
            # We maintain that behavior for OpenWakeWord default.
            config_model_path = self.config.get("chosen_wake_word_model_path")
            default_oww_model_path = config_model_path if engine_to_use == "openwakeword" and config_model_path else os.environ.get("WAKE_WORD_MODEL_PATH", "./hey_jimmy.onnx")
            if not default_oww_model_path or not isinstance(default_oww_model_path, str) or not os.path.exists(default_oww_model_path):
                print(f"CRITICAL: OpenWakeWord model '{default_oww_model_path}' not found. Please set WAKE_WORD_MODEL_PATH, configure via setup, or ensure the default model exists.")
                sys.exit(1)
            vc_args['engine_type'] = "openwakeword"
            vc_args['openwakeword_model_path'] = default_oww_model_path
            # If chosen_wake_word_engine was 'openwakeword' but model_path was None, this ensures it uses the env/default.

        try:
            self.core = VoiceCore(**vc_args)
        except ValueError as ve:
            print(f"CRITICAL: Failed to initialize VoiceCore: {ve}. Please check your configuration and environment variables.")
            sys.exit(1)
        except Exception as e_vc:
            print(f"CRITICAL: An unexpected error occurred during VoiceCore initialization: {e_vc}")
            sys.exit(1)

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
                    if spec is None:
                        print(f"  - FAILED to load '{filename}': Could not create module spec.")
                        continue
                    module = importlib.util.module_from_spec(spec)
                    if spec.loader is None:
                        print(f"  - FAILED to load '{filename}': No loader in module spec.")
                        continue
                    spec.loader.exec_module(module)
                    if hasattr(module, "register_intents"):
                        intents_from_module = module.register_intents()
                        self.intents.update(intents_from_module)
                        print(f"  - Loaded intents from '{filename}'")
                except Exception as e:
                    print(f"  - FAILED to load '{filename}': {e}")
        print(f"Total intents loaded: {len(self.intents)}")

    def handle_wake_word(self):
        """Called by the core when the wake word is detected."""
        print("\nAssistant: Wake word acknowledged. Listening for command...")
        speak("Yes?")

    def handle_command(self, command: str):
        """
        Called by the core with the transcribed command.
        Finds and executes the appropriate action from the loaded intents.
        """
        print(f"Assistant: Received command -> '{command}'")
        command = command.lower().strip()

        matched_intent = False # Flag to track if any intent matched
        # Priority 1: Exact match (typically for no-argument commands)
        for intent_phrase, action in self.intents.items():
            if command == intent_phrase:
                print(f"Assistant: Matched exact intent '{intent_phrase}'. Executing action.")
                try:
                    action() # Call with no arguments
                except TypeError as te:
                    # This happens if 'action' expected an argument but intent was registered as exact phrase.
                    print(f"ERROR: Action for '{intent_phrase}' likely expected an argument but received none: {te}")
                    speak("There was a configuration error for that command.")
                except Exception as e_action:
                    print(f"ERROR: Exception during action for '{intent_phrase}': {e_action}")
                    speak("Sorry, I encountered an error trying to do that.")
                matched_intent = True
                return # Exit once an exact match is handled

        # Priority 2: Starts-with match (for commands expecting one argument after the phrase)
        # Ensure intent_phrase is not empty to avoid issues with startswith(" ") if intent_phrase was ""
        if not matched_intent: # Only proceed if no exact match was found
            for intent_phrase, action in self.intents.items():
                if intent_phrase and command.startswith(intent_phrase + " "):
                    argument = command[len(intent_phrase):].strip() # Extract argument
                    if argument: # Ensure there is an actual argument, not just whitespace
                        print(f"Assistant: Matched keyword intent '{intent_phrase}' with argument '{argument}'. Executing action.")
                        try:
                            action(argument) # Call with one argument
                        except TypeError as te:
                            # This happens if 'action' did not expect an argument, or wrong number of args.
                            print(f"ERROR: Action for '{intent_phrase}' could not accept argument '{argument}': {te}")
                            # Check if it was perhaps an exact match that got missed, or if it should have been no-arg
                            # This might indicate an issue with intent registration or the command structure
                            # For now, assume it's a configuration error for this intent.
                            speak("There was a configuration error for that command type.")
                        except Exception as e_action_arg:
                            print(f"ERROR: Exception during action for '{intent_phrase}' with argument '{argument}': {e_action_arg}")
                            speak("Sorry, I encountered an error trying to perform that action.")
                        matched_intent = True
                        return # Exit once a keyword match is handled

        # If no intent was matched by either method
        if not matched_intent:
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
            if hasattr(tts_engine, 'stop'):
                tts_engine.stop() # Ensure tts_engine has stop
            if hasattr(self.core, 'stop'):
                self.core.stop()   # Ensure core has stop


if __name__ == "__main__":
    assistant = Assistant()
    assistant.run()
