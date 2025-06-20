import pytest
from unittest.mock import patch, AsyncMock, mock_open
import os
import sys
from datetime import datetime
import numpy as np  # Only import here for the test that needs it

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.intent_logic import (
    intent_handler,
    INTENT_HANDLERS,
    get_response,
    ShutdownSignal,
    process_command,
    handle_greeting_intent,
    handle_goodbye_intent,
    handle_set_reminder_intent,
    handle_list_reminders_intent,
    handle_get_weather_intent,
    handle_start_chat_with_llm,
    handle_retrain_model_intent
)

@pytest.fixture
def mock_response_map():
    """Mock response map for testing"""
    return {
        "greeting": "Hello! How can I help you today?",
        "goodbye": "Goodbye! Have a great day!",
        "set_reminder_error": "I couldn't set that reminder. Please try again.",
        "reminder_set_full": "Reminder set: {task} at {time}",
        "get_weather_current": "Current weather in {city}: {description}, {temp}°C",
        "llm_service_error": "Sorry, I'm having trouble processing that right now.",
        "retrain_model": "Starting model retraining...",
        "retrain_model_error": "Retraining failed: {error}"
    }

@pytest.fixture
def mock_entities():
    """Mock entities extracted from user input"""
    return {
        "task": "call mom",
        "time_phrase": "tomorrow at 3pm",
        "location": "New York"
    }

@pytest.fixture(autouse=True)
def setup_and_teardown():
    """Setup and teardown for each test"""
    # Clear the intent handlers registry before each test
    original_handlers = INTENT_HANDLERS.copy()
    yield
    # Restore original handlers after test
    INTENT_HANDLERS.clear()
    INTENT_HANDLERS.update(original_handlers)

class TestIntentHandlerDecorator:
    """Test the intent_handler decorator and registry system"""

    def test_intent_handler_decorator_registers_function(self):
        """Test that the intent_handler decorator properly registers functions"""
        @intent_handler("test_intent")
        async def test_function(text, entities):
            return "test response"

        assert "test_intent" in INTENT_HANDLERS
        assert INTENT_HANDLERS["test_intent"] == test_function

    def test_intent_handler_decorator_returns_original_function(self):
        """Test that the decorator returns the original function unchanged"""
        async def original_function(text, entities):
            return "original"

        decorated = intent_handler("test_intent2")(original_function)
        assert decorated == original_function

    def test_multiple_intent_handlers_registered(self):
        """Test that multiple intent handlers can be registered"""
        @intent_handler("intent1")
        async def handler1(text, entities):
            return "handler1"

        @intent_handler("intent2")
        async def handler2(text, entities):
            return "handler2"

        assert len(INTENT_HANDLERS) >= 2
        assert "intent1" in INTENT_HANDLERS
        assert "intent2" in INTENT_HANDLERS

class TestGetResponse:
    """Test the get_response function"""

    @patch('modules.intent_logic.RESPONSE_MAP')
    def test_get_response_basic_lookup(self, mock_response_map):
        """Test basic response lookup without formatting"""
        mock_response_map.get.return_value = "Hello World"
        result = get_response("greeting")
        assert result == "Hello World"
        mock_response_map.get.assert_called_once_with("greeting", "")

    @patch('modules.intent_logic.RESPONSE_MAP')
    def test_get_response_with_formatting(self, mock_response_map):
        """Test response with string formatting"""
        mock_response_map.get.return_value = "Hello {name}, the weather is {temp}°C"
        result = get_response("weather", name="John", temp=25)
        assert result == "Hello John, the weather is 25°C"

    @patch('modules.intent_logic.RESPONSE_MAP')
    def test_get_response_missing_key_in_format(self, mock_response_map):
        """Test handling of missing keys in format string"""
        mock_response_map.get.return_value = "Hello {name}, today is {day}"
        with patch('modules.intent_logic.logger') as mock_logger:
            result = get_response("greeting", name="John")  # missing 'day'
            assert result == "Hello {name}, today is {day}"
            mock_logger.warning.assert_called_once()

    @patch('modules.intent_logic.RESPONSE_MAP')
    def test_get_response_formatting_error(self, mock_response_map):
        """Test handling of other formatting errors"""
        mock_response_map.get.return_value = "Value: {value:invalid_format}" # type: ignore
        with patch('modules.intent_logic.logger') as mock_logger:
            result = get_response("test", value="test")
            assert result == "Value: {value:invalid_format}"
            mock_logger.warning.assert_called_once()

    @patch('modules.intent_logic.RESPONSE_MAP')
    def test_get_response_nonexistent_intent(self, mock_response_map):
        """Test response for non-existent intent"""
        mock_response_map.get.return_value = ""
        result = get_response("nonexistent_intent")
        assert result == ""

class TestIntentHandlers:
    """Test individual intent handler functions"""

    @patch('modules.intent_logic.text_to_speech_async')
    @patch('modules.intent_logic.get_response')
    @pytest.mark.asyncio
    async def test_handle_greeting_intent_success(self, mock_get_response, mock_tts):
        """Test successful greeting intent handling"""
        mock_get_response.return_value = "Hello there!"
        mock_tts.return_value = None

        result = await handle_greeting_intent("hello", {})

        assert result == "Hello there!"
        mock_get_response.assert_called_once_with("greeting")
        mock_tts.assert_called_once_with("Hello there!")

    @patch('modules.intent_logic.text_to_speech_async')
    @patch('modules.intent_logic.get_response')
    @pytest.mark.asyncio
    async def test_handle_goodbye_intent_raises_shutdown_signal(self, mock_get_response, mock_tts):
        """Test that goodbye intent raises ShutdownSignal"""
        mock_get_response.side_effect = ["Goodbye!", "Shutting down assistant as requested by user."]
        mock_tts.return_value = None

        with pytest.raises(ShutdownSignal, match="User requested shutdown"):
            await handle_goodbye_intent("goodbye", {})

        assert mock_tts.call_count == 2
        assert mock_get_response.call_count == 1

    @patch('modules.intent_logic.text_to_speech_async')
    @patch('modules.intent_logic.get_response')
    @patch('modules.intent_logic.run_validation_and_retrain_async')
    @pytest.mark.asyncio
    async def test_handle_retrain_model_intent_success(self, mock_retrain, mock_get_response, mock_tts):
        """Test successful model retraining"""
        mock_get_response.return_value = "Starting retraining..."
        mock_retrain.return_value = (True, "Retraining completed successfully")
        mock_tts.return_value = None

        result = await handle_retrain_model_intent("retrain model", {})

        assert result == "Starting retraining..."
        mock_retrain.assert_called_once()
        assert mock_tts.call_count == 2

    @patch('modules.intent_logic.text_to_speech_async')
    @patch('modules.intent_logic.get_response')
    @patch('modules.intent_logic.run_validation_and_retrain_async')
    @pytest.mark.asyncio
    async def test_handle_retrain_model_intent_failure(self, mock_retrain, mock_get_response, mock_tts):
        """Test model retraining failure"""
        mock_get_response.side_effect = ["Starting retraining...", "Retraining failed: Test error"]
        mock_retrain.side_effect = Exception("Test error")
        mock_tts.return_value = None

        result = await handle_retrain_model_intent("retrain model", {})

        assert result == "Starting retraining..."
        mock_retrain.assert_called_once()
        assert mock_get_response.call_count == 2

    @patch('modules.intent_logic.text_to_speech_async')
    @patch('modules.intent_logic.get_response')
    @patch('modules.intent_logic.save_reminder_async')
    @patch('modules.intent_logic.add_event_to_calendar')
    @patch('modules.intent_logic.dateparser')
    @pytest.mark.asyncio
    async def test_handle_set_reminder_intent_with_entities(self, mock_dateparser, mock_calendar, mock_save, mock_get_response, mock_tts):
        """Test setting reminder with complete entities"""
        mock_datetime = datetime(2024, 12, 25, 15, 0)
        mock_dateparser.parse.return_value = mock_datetime
        mock_get_response.side_effect = [
            "Reminder set: call mom at 03:00 PM on Wednesday, December 25",
            "Added to calendar"
        ]
        mock_save.return_value = None
        mock_calendar.return_value = None
        mock_tts.return_value = None

        entities = {"task": "call mom", "time_phrase": "tomorrow at 3pm"}
        result = await handle_set_reminder_intent("set reminder to call mom tomorrow at 3pm", entities)

        mock_save.assert_called_once_with("call mom", mock_datetime)
        mock_calendar.assert_called_once_with("call mom", mock_datetime)
        mock_tts.assert_called_once()
        assert "call mom" in result

    @patch('modules.intent_logic.text_to_speech_async')
    @patch('modules.intent_logic.get_response')
    @patch('modules.intent_logic.get_reminders_for_date_async')
    @patch('modules.intent_logic.parse_list_reminder_request')
    @patch('modules.intent_logic.show_reminders_gui')
    @patch('modules.intent_logic.threading.Thread')
    @pytest.mark.asyncio
    async def test_handle_list_reminders_intent_with_reminders(self, mock_thread, mock_gui, mock_parse_request, mock_get_reminders, mock_get_response, mock_tts):
        """Test listing reminders when reminders exist"""
        target_date = datetime(2024, 12, 25)
        mock_parse_request.return_value = target_date
        mock_reminders = [
            {"task": "call mom", "time": datetime(2024, 12, 25, 15, 0)},
            {"task": "buy groceries", "time": datetime(2024, 12, 25, 18, 0)}
        ]
        mock_get_reminders.return_value = mock_reminders
        mock_get_response.return_value = (
            "Your reminders for Tuesday, December 25, 2024: "
            "call mom at 03:00 PM. buy groceries at 06:00 PM."
        )
        mock_tts.return_value = None

        result = await handle_list_reminders_intent("list reminders for today", {})

        mock_get_reminders.assert_called_once_with(target_date)
        mock_tts.assert_called_once()
        mock_thread.assert_called_once()
        assert "call mom" in result

    @patch('modules.intent_logic.text_to_speech_async')
    @patch('modules.intent_logic.get_response')
    @patch('modules.intent_logic.get_weather_async')
    @pytest.mark.asyncio
    async def test_handle_get_weather_intent_current_location(self, mock_weather, mock_get_response, mock_tts):
        """Test getting weather for current location"""
        mock_weather_data = {
            "city": "New York",
            "description": "sunny",
            "temp": 25.0
        }
        mock_weather.return_value = mock_weather_data
        mock_get_response.return_value = "Current weather in New York: sunny, 25.0°C"
        mock_tts.return_value = None

        entities = {"location": "current"}
        result = await handle_get_weather_intent("what's the weather", entities)

        mock_weather.assert_called_once_with(location_query=None, entities=entities)
        mock_tts.assert_called()  # Called twice - once for "fetching" message
        assert "New York" in result

class TestProcessCommand:
    """Test the main process_command function"""

    @patch('modules.intent_logic.normalize_text')
    @patch('modules.intent_logic.detect_intent_async')
    @patch('modules.intent_logic.handle_goodbye_intent')
    @pytest.mark.asyncio
    async def test_process_command_goodbye_heuristic(self, mock_goodbye_handler, mock_detect_intent, mock_normalize):
        """Test that goodbye heuristic triggers even with different intent detection"""
        mock_normalize.return_value = "goodbye friend"
        mock_detect_intent.return_value = ("unknown", {})
        mock_goodbye_handler.return_value = "Goodbye!"

        await process_command("goodbye friend")
        mock_goodbye_handler.assert_called_once_with("goodbye friend", {})

    @patch('modules.intent_logic.normalize_text')
    @patch('modules.intent_logic.detect_intent_async')
    @patch('modules.intent_logic.parse_retrain_request')
    @patch('modules.intent_logic.run_validation_and_retrain_async')
    @patch('modules.intent_logic.text_to_speech_async')
    @patch('modules.intent_logic.get_response')
    @pytest.mark.asyncio
    async def test_process_command_retrain_model_special_handling(self, mock_get_response, mock_tts, mock_retrain, mock_parse_retrain, mock_detect_intent, mock_normalize):
        """Test special handling for retrain_model intent"""
        mock_normalize.return_value = "retrain the model"
        mock_detect_intent.return_value = ("retrain_model", {})
        mock_parse_retrain.return_value = True
        mock_retrain.return_value = (True, "Retraining successful")
        mock_get_response.return_value = "Starting retraining..."
        mock_tts.return_value = None

        await process_command("retrain the model")
        mock_retrain.assert_called_once()
        assert mock_tts.call_count == 2  # initial and retrain messages

    @patch('modules.intent_logic.normalize_text')
    @patch('modules.intent_logic.detect_intent_async')
    @patch('modules.intent_logic.get_llm_response')
    @patch('modules.intent_logic.text_to_speech_async')
    @patch('modules.intent_logic.get_response')
    @pytest.mark.asyncio
    async def test_process_command_fallback_to_llm(self, mock_get_response, mock_tts, mock_llm, mock_detect_intent, mock_normalize):
        """Test fallback to LLM for unhandled intents"""
        mock_normalize.return_value = "what is the meaning of life"
        mock_detect_intent.return_value = ("unknown", {})
        mock_llm.return_value = "42"
        mock_tts.return_value = None

        await process_command("what is the meaning of life")
        mock_llm.assert_called_once_with(input_text="what is the meaning of life")
        mock_tts.assert_called_once_with("42")

    @patch('modules.intent_logic.normalize_text')
    @patch('modules.intent_logic.detect_intent_async')
    @patch('modules.intent_logic.get_llm_response')
    @patch('modules.intent_logic.text_to_speech_async')
    @patch('modules.intent_logic.get_response')
    @pytest.mark.asyncio
    async def test_process_command_llm_service_error(self, mock_get_response, mock_tts, mock_llm, mock_detect_intent, mock_normalize):
        """Test handling when LLM service returns None"""
        mock_normalize.return_value = "test input"
        mock_detect_intent.return_value = ("unknown", {})
        mock_llm.return_value = None
        mock_get_response.return_value = "LLM service error"
        mock_tts.return_value = None

        await process_command("test input")
        mock_get_response.assert_called_with("llm_service_error")
        mock_tts.assert_called_with("LLM service error")

    @patch('modules.intent_logic.text_to_speech_async')
    @patch('modules.intent_logic.get_response')
    @patch('modules.intent_logic.record_audio_async', new_callable=AsyncMock)
    @patch('modules.intent_logic.transcribe_audio_async')
    @patch('modules.intent_logic.get_llm_response')
    @patch('modules.intent_logic.json.dump')
    @patch('modules.intent_logic.open', new_callable=mock_open)
    @patch('modules.intent_logic.os.makedirs')
    @pytest.mark.asyncio
    async def test_handle_start_chat_with_llm_successful_conversation(
        self, mock_makedirs, mock_file, mock_json_dump,
        mock_llm, mock_transcribe, mock_record,
        mock_get_response, mock_tts
    ):
        """Test successful chat session with LLM"""
        mock_get_response.side_effect = ["Starting chat mode...", "Chat session saved successfully"]
        mock_record.side_effect = [
            np.array([1, 2, 3], dtype=np.int16),  # Return np.ndarray
            np.array([4, 5, 6], dtype=np.int16),  # Return np.ndarray
            np.array([7, 8, 9], dtype=np.int16)   # Return np.ndarray for the stop phrase
        ]
        mock_transcribe.side_effect = [
            "Hello there",
            "How are you?",
            "stop chat and save"
        ]
        mock_llm.side_effect = [
            "Hello! I'm doing well.",
            "I'm great, thanks for asking!"
        ]
        mock_tts.return_value = None

        result = await handle_start_chat_with_llm("start chat", {})
        assert mock_record.call_count == 3
        assert mock_transcribe.call_count == 3
        assert mock_llm.call_count == 2
        mock_json_dump.assert_called_once()
        assert result == "Chat session saved successfully"

class TestParametrizedScenarios:
    """Parametrized tests for various scenarios"""

    @pytest.mark.parametrize("intent,expected_handler", [
        ("greeting", "handle_greeting_intent"),
        ("goodbye", "handle_goodbye_intent"),
        ("set_reminder", "handle_set_reminder_intent"),
        ("list_reminders", "handle_list_reminders_intent"),
        ("get_weather", "handle_get_weather_intent"),
    ])
    def test_intent_handlers_registered(self, intent, expected_handler):
        """Test that all expected intent handlers are registered"""
        assert intent in INTENT_HANDLERS
        assert INTENT_HANDLERS[intent].__name__ == expected_handler

    @pytest.mark.parametrize("goodbye_phrase", [
        "goodbye",
        "bye",
        "see you later",
        "farewell",
        "exit",
        "quit",
        "terminate"
    ]) # type: ignore
    @patch('modules.intent_logic.normalize_text')
    @patch('modules.intent_logic.detect_intent_async')
    @patch('modules.intent_logic.handle_goodbye_intent')
    @pytest.mark.asyncio
    async def test_goodbye_heuristic_phrases(self, mock_goodbye_handler, mock_detect_intent, mock_normalize, goodbye_phrase):
        """Test that various goodbye phrases trigger the heuristic"""
        mock_normalize.return_value = goodbye_phrase
        mock_detect_intent.return_value = ("unknown", {})
        mock_goodbye_handler.return_value = "Goodbye!"

        await process_command(goodbye_phrase)
        mock_goodbye_handler.assert_called_once()

class TestErrorHandling:
    """Test error handling and edge cases"""

    @pytest.mark.asyncio
    async def test_shutdown_signal_exception(self):
        """Test that ShutdownSignal can be raised and caught"""
        with pytest.raises(ShutdownSignal):
            raise ShutdownSignal("Test shutdown")

    @patch('modules.intent_logic.text_to_speech_async')
    @patch('modules.intent_logic.get_response')
    @pytest.mark.asyncio
    async def test_intent_handler_with_tts_failure(self, mock_get_response, mock_tts):
        """Test intent handler behavior when TTS fails"""
        mock_get_response.return_value = "Hello!"
        mock_tts.side_effect = Exception("TTS failed")

        with pytest.raises(Exception, match="TTS failed"):
            await handle_greeting_intent("hello", {})

    @patch('modules.intent_logic.normalize_text')
    @patch('modules.intent_logic.detect_intent_async')
    @pytest.mark.asyncio
    async def test_process_command_with_detection_failure(self, mock_detect_intent, mock_normalize):
        """Test process_command when intent detection fails"""
        mock_normalize.return_value = "test input"
        mock_detect_intent.side_effect = Exception("Detection failed")

        with patch('modules.intent_logic.logger'):
            await process_command("test input")
            # Error should be caught by decorator and logged

if __name__ == "__main__":
    pytest.main([__file__])
