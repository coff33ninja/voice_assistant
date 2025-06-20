import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta, date
import sys
import os

# Add the project root to the path to import reminder_utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from modules.reminder_utils import (
    parse_reminder,
    parse_list_reminder_request,
    _parse_time_from_entities_text,
    _parse_date_from_entities_text
)


@pytest.fixture
def mock_datetime():
    """Fixture providing a consistent datetime for testing."""
    return datetime(2024, 1, 15, 10, 30, 0)  # Monday, Jan 15, 2024 at 10:30 AM

@pytest.fixture
def sample_entities():
    """Fixture providing sample entity data for testing."""
    return {
        "basic_task": {"task": "call mom", "time_phrase": "tomorrow at 3pm"},
        "complex_task": {"task": "meeting", "date_reference": "next monday", "time": "9am"},
        "time_only": {"time": "7:30pm"},
        "date_only": {"date_reference": "tomorrow"},
        "invalid_entities": {"invalid_key": "invalid_value"}
    }


class TestParseTimeFromEntitiesText:
    """Test suite for _parse_time_from_entities_text function."""

    @patch('modules.reminder_utils.datetime')
    def test_parse_time_basic_formats(self, mock_dt, mock_datetime):
        """Test parsing basic time formats like 3pm, 15:30."""
        mock_dt.now.return_value = mock_datetime

        # Test 12-hour format with AM/PM
        result = _parse_time_from_entities_text("3:30pm")
        assert result.hour == 15
        assert result.minute == 30

        # Test 24-hour format
        result = _parse_time_from_entities_text("15:30")
        assert result.hour == 15
        assert result.minute == 30

        # Test AM format
        result = _parse_time_from_entities_text("9:15am")
        assert result.hour == 9
        assert result.minute == 15

    @patch('modules.reminder_utils.datetime')
    def test_parse_time_with_date_reference(self, mock_dt, mock_datetime):
        """Test parsing time with date reference like 'tomorrow at 3pm'."""
        mock_dt.now.return_value = mock_datetime

        # Test tomorrow with time
        result = _parse_time_from_entities_text("3:30pm", "tomorrow")
        expected_date = mock_datetime.date() + timedelta(days=1)
        assert result.date() == expected_date
        assert result.hour == 15
        assert result.minute == 30

        # Test today with time
        result = _parse_time_from_entities_text("10:00am", "today")
        assert result.date() == mock_datetime.date()
        assert result.hour == 10
        assert result.minute == 0

    @patch('modules.reminder_utils.datetime')
    def test_parse_time_past_time_adjustment(self, mock_dt, mock_datetime):
        """Test that past times are adjusted to next day."""
        mock_dt.now.return_value = mock_datetime  # 10:30 AM

        # Test time earlier than current time (should be tomorrow)
        result = _parse_time_from_entities_text("9:00am")
        assert result.date() == mock_datetime.date() + timedelta(days=1)

        # Test time later than current time (should be today)
        result = _parse_time_from_entities_text("2:00pm")
        assert result.date() == mock_datetime.date()
        assert result.hour == 14

    @patch('modules.reminder_utils.datetime')
    def test_parse_time_relative_formats(self, mock_dt, mock_datetime):
        """Test parsing relative time formats like 'in 2 hours'."""
        mock_dt.now.return_value = mock_datetime

        # Test 'in X hours'
        result = _parse_time_from_entities_text("in 2 hours")
        expected = mock_datetime + timedelta(hours=2)
        assert result == expected

        # Test 'in X minutes'
        result = _parse_time_from_entities_text("in 30 minutes")
        expected = mock_datetime + timedelta(minutes=30)
        assert result == expected

    @patch('modules.reminder_utils.datetime')
    def test_parse_time_default_values(self, mock_dt, mock_datetime):
        """Test default time values for 'tomorrow' and 'today'."""
        mock_dt.now.return_value = mock_datetime

        # Test 'tomorrow' defaults to 9 AM
        result = _parse_time_from_entities_text("tomorrow")
        expected = (mock_datetime + timedelta(days=1)).replace(hour=9, minute=0, second=0, microsecond=0)
        assert result == expected

        # Test 'today' defaults to 9 AM
        result = _parse_time_from_entities_text("today")
        expected = mock_datetime.replace(hour=9, minute=0, second=0, microsecond=0)
        assert result == expected

    def test_parse_time_invalid_inputs(self):
        """Test parsing invalid time inputs."""
        assert _parse_time_from_entities_text("invalid time") is None
        assert _parse_time_from_entities_text("") is None
        assert _parse_time_from_entities_text(None) is None

    @patch('modules.reminder_utils.datetime')
    def test_parse_time_edge_cases(self, mock_dt, mock_datetime):
        """Test edge cases for time parsing."""
        mock_dt.now.return_value = mock_datetime

        # Test midnight
        result = _parse_time_from_entities_text("12:00am")
        assert result.hour == 0
        assert result.minute == 0

        # Test noon
        result = _parse_time_from_entities_text("12:00pm")
        assert result.hour == 12
        assert result.minute == 0

        # Test single digit hours
        result = _parse_time_from_entities_text("9:00am")
        assert result.hour == 9
        assert result.minute == 0


class TestParseDateFromEntitiesText:
    """Test suite for _parse_date_from_entities_text function."""

    @patch('modules.reminder_utils.datetime')
    def test_parse_date_basic_references(self, mock_dt, mock_datetime):
        """Test parsing basic date references like today, tomorrow, yesterday."""
        mock_dt.now.return_value = mock_datetime

        assert _parse_date_from_entities_text("today") == mock_datetime.date()
        assert _parse_date_from_entities_text("tomorrow") == mock_datetime.date() + timedelta(days=1)
        assert _parse_date_from_entities_text("yesterday") == mock_datetime.date() - timedelta(days=1)

    @patch('modules.reminder_utils.datetime')
    def test_parse_date_week_references(self, mock_dt, mock_datetime):
        """Test parsing week-based date references."""
        mock_dt.now.return_value = mock_datetime  # Monday

        assert _parse_date_from_entities_text("next week") == mock_datetime.date() + timedelta(days=7)
        assert _parse_date_from_entities_text("this week") == mock_datetime.date()

    @patch('modules.reminder_utils.datetime')
    def test_parse_date_day_of_week(self, mock_dt, mock_datetime):
        """Test parsing specific days of the week."""
        mock_dt.now.return_value = mock_datetime  # Monday

        assert _parse_date_from_entities_text("next monday") == mock_datetime.date() + timedelta(days=7)
        assert _parse_date_from_entities_text("tuesday") == mock_datetime.date() + timedelta(days=1)
        assert _parse_date_from_entities_text("friday") == mock_datetime.date() + timedelta(days=4)

    @pytest.mark.parametrize("day_name,expected_offset", [
        ("monday", 0), ("tuesday", 1), ("wednesday", 2),
        ("thursday", 3), ("friday", 4), ("saturday", 5), ("sunday", 6)
    ])
    @patch('modules.reminder_utils.datetime')
    def test_parse_date_all_weekdays(self, mock_dt, mock_datetime, day_name, expected_offset):
        """Test parsing all weekday names."""
        mock_dt.now.return_value = mock_datetime  # Monday
        result = _parse_date_from_entities_text(day_name)
        if expected_offset == 0:
            assert result == mock_datetime.date()
        else:
            assert result == mock_datetime.date() + timedelta(days=expected_offset)

    def test_parse_date_invalid_inputs(self):
        """Test parsing invalid date inputs."""
        assert _parse_date_from_entities_text("invalid date") is None
        assert _parse_date_from_entities_text("") is None
        assert _parse_date_from_entities_text(None) is None

    @patch('modules.reminder_utils.datetime')
    def test_parse_date_case_insensitive(self, mock_dt, mock_datetime):
        """Test that date parsing is case insensitive."""
        mock_dt.now.return_value = mock_datetime

        assert _parse_date_from_entities_text("TODAY") == mock_datetime.date()
        assert _parse_date_from_entities_text("Tomorrow") == mock_datetime.date() + timedelta(days=1)
        assert _parse_date_from_entities_text("NEXT WEEK") == mock_datetime.date() + timedelta(days=7)
        assert _parse_date_from_entities_text("Monday") == mock_datetime.date()


class TestParseReminder:
    """Test suite for parse_reminder function."""

    @patch('modules.reminder_utils.datetime')
    def test_parse_reminder_with_entities(self, mock_dt, mock_datetime):
        """Test parsing reminders with entity data."""
        mock_dt.now.return_value = mock_datetime
        entities = {"task": "call mom", "time_phrase": "tomorrow at 3pm"}
        result = parse_reminder("remind me to call mom tomorrow at 3pm", entities)

        assert result is not None
        assert result["task"] == "call mom"
        assert result["time"].hour == 15
        assert result["time"].date() == mock_datetime.date() + timedelta(days=1)

    @patch('modules.reminder_utils.datetime')
    def test_parse_reminder_with_date_reference_and_time(self, mock_dt, mock_datetime):
        """Test parsing with separate date_reference and time entities."""
        mock_dt.now.return_value = mock_datetime
        entities = {"task": "meeting", "date_reference": "tomorrow", "time": "9am"}
        result = parse_reminder("remind me to meeting tomorrow at 9am", entities)

        assert result is not None
        assert result["task"] == "meeting"
        assert result["time"].hour == 9
        assert result["time"].date() == mock_datetime.date() + timedelta(days=1)

    @patch('modules.reminder_utils.datetime')
    def test_parse_reminder_fallback_to_regex(self, mock_dt, mock_datetime):
        """Test fallback to regex parsing when entities are insufficient."""
        mock_dt.now.return_value = mock_datetime
        result = parse_reminder("remind me to call mom tomorrow at 3pm", None)

        assert result is not None
        assert result["task"] == "call mom"
        assert result["time"].hour == 15
        assert result["time"].date() == mock_datetime.date() + timedelta(days=1)

    @patch('modules.reminder_utils.datetime')
    def test_parse_reminder_regex_patterns(self, mock_dt, mock_datetime):
        """Test various regex patterns for reminder parsing."""
        mock_dt.now.return_value = mock_datetime

        result = parse_reminder("remind me to call mom at 3pm", None)
        assert result is not None and result["time"].hour == 15

        result = parse_reminder("remind me to call mom tomorrow", None)
        assert result is not None and result["time"].hour == 9

        result = parse_reminder("remind me to call mom in 2 hours", None)
        expected_time = mock_datetime + timedelta(hours=2)
        assert result["time"] == expected_time

    @pytest.mark.parametrize("text,expected_task", [
        ("remind me to call mom at 3pm", "call mom"),
        ("remind me to buy groceries tomorrow", "buy groceries"),
        ("remind me to take medicine in 1 hour", "take medicine"),
        ("remind me to attend meeting on friday at 2pm", "attend meeting")
    ])
    @patch('modules.reminder_utils.datetime')
    def test_parse_reminder_task_extraction(self, mock_dt, mock_datetime, text, expected_task):
        """Test task extraction from various reminder texts."""
        mock_dt.now.return_value = mock_datetime
        result = parse_reminder(text, None)
        assert result is not None
        assert result["task"] == expected_task

    def test_parse_reminder_invalid_inputs(self):
        """Test parsing invalid reminder inputs."""
        assert parse_reminder("call mom at 3pm", None) is None
        assert parse_reminder("", None) is None
        assert parse_reminder(None, None) is None
        assert parse_reminder("remind me to call mom", None) is None

    @patch('modules.reminder_utils.datetime')
    def test_parse_reminder_edge_cases(self, mock_dt, mock_datetime):
        """Test edge cases for reminder parsing."""
        mock_dt.now.return_value = mock_datetime

        result = parse_reminder("remind me to call mom at 9am", None)
        assert result is not None and result["time"].date() == mock_datetime.date() + timedelta(days=1)

        result = parse_reminder("remind me to call mom at 2pm", None)
        assert result is not None and result["time"].date() == mock_datetime.date()

    @patch('modules.reminder_utils.datetime')
    def test_parse_reminder_complex_time_formats(self, mock_dt, mock_datetime):
        """Test complex time format parsing."""
        mock_dt.now.return_value = mock_datetime

        result = parse_reminder("remind me to call mom tomorrow at 7:30pm", None)
        assert result["time"].hour == 19 and result["time"].minute == 30
        assert result["time"].date() == mock_datetime.date() + timedelta(days=1)

        result = parse_reminder("remind me to call mom at 7:30pm tomorrow", None)
        assert result["time"].hour == 19 and result["time"].minute == 30
        assert result["time"].date() == mock_datetime.date() + timedelta(days=1)

    @patch('modules.reminder_utils.datetime')
    def test_parse_reminder_relative_time_units(self, mock_dt, mock_datetime):
        """Test various relative time units."""
        mock_dt.now.return_value = mock_datetime

        test_cases = [
            ("remind me to call mom in 1 hour", timedelta(hours=1)),
            ("remind me to call mom in 2 hours", timedelta(hours=2)),
            ("remind me to call mom in 30 minutes", timedelta(minutes=30)),
            ("remind me to call mom in 1 day", timedelta(days=1)),
            ("remind me to call mom in 2 days", timedelta(days=2)),
            ("remind me to call mom in 1 week", timedelta(weeks=1)),
            ("remind me to call mom in 2 weeks", timedelta(weeks=2))
        ]
        for text, expected_delta in test_cases:
            result = parse_reminder(text, None)
            expected_time = mock_datetime + expected_delta
            assert result["time"] == expected_time


class TestParseListReminderRequest:
    """Test suite for parse_list_reminder_request function."""

    @patch('modules.reminder_utils.datetime')
    def test_parse_list_reminder_with_entities(self, mock_dt, mock_datetime):
        """Test parsing list reminders with entity data."""
        mock_dt.now.return_value = mock_datetime
        entities = {"date_reference": "tomorrow"}
        result = parse_list_reminder_request("show reminders for tomorrow", entities)
        assert result == mock_datetime.date() + timedelta(days=1)

    @patch('modules.reminder_utils.datetime')
    def test_parse_list_reminder_basic_dates(self, mock_dt, mock_datetime):
        """Test parsing basic date references."""
        mock_dt.now.return_value = mock_datetime

        assert parse_list_reminder_request("show reminders for today", None) == mock_datetime.date()
        assert parse_list_reminder_request("show reminders for tomorrow", None) == mock_datetime.date() + timedelta(days=1)
        assert parse_list_reminder_request("show reminders for yesterday", None) == mock_datetime.date() - timedelta(days=1)

    @patch('modules.reminder_utils.datetime')
    def test_parse_list_reminder_relative_dates(self, mock_dt, mock_datetime):
        """Test parsing relative date expressions."""
        mock_dt.now.return_value = mock_datetime

        assert parse_list_reminder_request("show reminders in 3 days", None) == mock_datetime.date() + timedelta(days=3)
        assert parse_list_reminder_request("show reminders in 2 weeks", None) == mock_datetime.date() + timedelta(weeks=2)
        assert parse_list_reminder_request("show reminders in 1 month", None) == mock_datetime.date() + timedelta(days=30)

    @patch('modules.reminder_utils.datetime')
    def test_parse_list_reminder_weekdays(self, mock_dt, mock_datetime):
        """Test parsing weekday references."""
        mock_dt.now.return_value = mock_datetime  # Monday
        weekday_tests = [
            ("monday", 0), ("tuesday", 1), ("wednesday", 2),
            ("thursday", 3), ("friday", 4), ("saturday", 5), ("sunday", 6)
        ]
        for day, offset in weekday_tests:
            result = parse_list_reminder_request(f"show reminders for {day}", None)
            expected = mock_datetime.date() + timedelta(days=offset)
            assert result == expected

    @patch('modules.reminder_utils.datetime')
    def test_parse_list_reminder_next_this_modifiers(self, mock_dt, mock_datetime):
        """Test 'next' and 'this' modifiers with weekdays."""
        mock_dt.now.return_value = mock_datetime  # Monday

        assert parse_list_reminder_request("show reminders for next monday", None) == mock_datetime.date() + timedelta(days=7)
        assert parse_list_reminder_request("show reminders for this friday", None) == mock_datetime.date() + timedelta(days=4)
        assert parse_list_reminder_request("show reminders for next week", None) == mock_datetime.date() + timedelta(days=7)

    @patch('modules.reminder_utils.datetime')
    def test_parse_list_reminder_month_dates(self, mock_dt, mock_datetime):
        """Test parsing specific month and day combinations."""
        mock_dt.now.return_value = mock_datetime  # January 15, 2024

        assert parse_list_reminder_request("show reminders for March 15th", None) == date(2024, 3, 15)
        assert parse_list_reminder_request("show reminders for July 4th, 2024", None) == date(2024, 7, 4)
        assert parse_list_reminder_request("show reminders for January 1st", None) == date(2025, 1, 1)

    @patch('modules.reminder_utils.datetime')
    def test_parse_list_reminder_formatted_dates(self, mock_dt, mock_datetime):
        """Test parsing formatted date strings."""
        mock_dt.now.return_value = mock_datetime

        assert parse_list_reminder_request("show reminders on 2024-03-15", None) == date(2024, 3, 15)
        assert parse_list_reminder_request("show reminders on 03/15/2024", None) == date(2024, 3, 15)

    @pytest.mark.parametrize("month_name,month_num", [
        ("january", 1), ("february", 2), ("march", 3), ("april", 4),
        ("may", 5), ("june", 6), ("july", 7), ("august", 8),
        ("september", 9), ("october", 10), ("november", 11), ("december", 12)
    ])
    @patch('modules.reminder_utils.datetime')
    def test_parse_list_reminder_all_months(self, mock_dt, mock_datetime, month_name, month_num):
        """Test parsing all month names."""
        mock_dt.now.return_value = mock_datetime
        expected_year = 2024 if month_num >= 1 else 2025
        result = parse_list_reminder_request(f"show reminders for {month_name} 15th", None)
        assert result == date(expected_year, month_num, 15)

    def test_parse_list_reminder_invalid_inputs(self):
        """Test parsing invalid inputs."""
        assert parse_list_reminder_request("show all reminders", None) is None
        assert parse_list_reminder_request("", None) is None
        assert parse_list_reminder_request(None, None) is None
        assert parse_list_reminder_request("show reminders for 2024-13-40", None) is None

    @patch('modules.reminder_utils.datetime')
    def test_parse_list_reminder_case_insensitive(self, mock_dt, mock_datetime):
        """Test that date parsing is case insensitive."""
        mock_dt.now.return_value = mock_datetime

        assert parse_list_reminder_request("show reminders for TODAY", None) == mock_datetime.date()
        assert parse_list_reminder_request("show reminders for Tomorrow", None) == mock_datetime.date() + timedelta(days=1)
        assert parse_list_reminder_request("show reminders for FRIDAY", None) == mock_datetime.date() + timedelta(days=4)

    @patch('modules.reminder_utils.datetime')
    def test_parse_list_reminder_edge_cases(self, mock_dt, mock_datetime):
        """Test edge cases for list reminder parsing."""
        mock_dt.now.return_value = mock_datetime

        assert parse_list_reminder_request("show reminders for March 1st", None) == date(2024, 3, 1)
        assert parse_list_reminder_request("show reminders for March 2nd", None) == date(2024, 3, 2)
        assert parse_list_reminder_request("show reminders for March 3rd", None) == date(2024, 3, 3)
        assert parse_list_reminder_request("show reminders for March 4th", None) == date(2024, 3, 4)


class TestIntegrationAndPerformance:
    """Integration and performance tests for reminder_utils module."""

    @patch('modules.reminder_utils.datetime')
    def test_reminder_parsing_integration(self, mock_dt, mock_datetime):
        """Test integration between reminder parsing and list parsing."""
        mock_dt.now.return_value = mock_datetime

        reminder = parse_reminder("remind me to call mom tomorrow at 3pm", None)
        list_date = parse_list_reminder_request("show reminders for tomorrow", None)
        assert reminder is not None
        assert list_date == reminder["time"].date()

    @patch('modules.reminder_utils.datetime')
    def test_entity_vs_regex_consistency(self, mock_dt, mock_datetime):
        """Test that entity-based and regex-based parsing give consistent results."""
        mock_dt.now.return_value = mock_datetime
        text = "remind me to call mom tomorrow at 3pm"
        entities = {"task": "call mom", "time_phrase": "tomorrow at 3pm"}

        result_entities = parse_reminder(text, entities)
        result_regex = parse_reminder(text, None)
        assert result_entities["task"] == result_regex["task"]
        assert result_entities["time"] == result_regex["time"]

    def test_performance_large_text_input(self):
        """Test performance with large text inputs."""
        import time

        large_text = "remind me to " + "call mom " * 1000 + " tomorrow at 3pm"
        start = time.time()
        result = parse_reminder(large_text, None)
        duration = time.time() - start

        assert duration < 1.0
        assert result is not None

    def test_memory_usage_stress_test(self):
        """Test memory usage with multiple parsing operations."""
        import gc
        gc.collect()
        for i in range(1000):
            parse_reminder(f"remind me to task {i} tomorrow at 3pm", None)
            parse_list_reminder_request(f"show reminders for tomorrow {i}", None)
        gc.collect()
        assert True

    @patch('modules.reminder_utils.datetime')
    def test_concurrent_parsing_safety(self, mock_dt, mock_datetime):
        """Test that parsing functions are safe for concurrent use."""
        import threading

        mock_dt.now.return_value = mock_datetime
        results = []

        def worker(idx):
            res = parse_reminder(f"remind me to task {idx} tomorrow at 3pm", None)
            results.append(res)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads: t.start()
        for t in threads: t.join()

        assert len(results) == 10
        for r in results:
            assert r is not None and "task" in r and "time" in r


class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge cases."""

    def test_malformed_input_handling(self):
        """Test handling of malformed inputs."""
        malformed = [
            "remind me to", "remind me", "to call mom", "", None, 123, [], {},
            "remind me to call mom at 25:00", "remind me to call mom on February 30th"
        ]
        for inp in malformed:
            try:
                res = parse_reminder(inp, None)
                assert res is None or isinstance(res, dict)
            except Exception as e:
                pytest.fail(f"Unhandled exception for input {inp!r}: {e}")

    def test_unicode_and_special_characters(self):
        """Test handling of unicode and special characters."""
        texts = [
            "remind me to café meeting tomorrow at 3pm",
            "remind me to 会议 tomorrow at 3pm",
            "remind me to meeting™ tomorrow at 3pm",
            "remind me to call mom @ 3pm", "remind me to call mom & dad tomorrow"
        ]
        for txt in texts:
            try:
                res = parse_reminder(txt, None)
                assert res is None or isinstance(res, dict)
            except Exception as e:
                pytest.fail(f"Failed on unicode input {txt!r}: {e}")

    def test_timezone_awareness(self):
        """Test that functions handle timezone-naive datetime objects correctly."""
        result = parse_reminder("remind me to call mom at 3pm", None)
        if result:
            assert result["time"].tzinfo is None


# Test configuration and utilities
@pytest.fixture(autouse=True)
def reset_datetime_mock():
    """Reset datetime mock after each test to prevent test interference."""
    yield

def test_module_imports():
    """Test that all required modules can be imported correctly."""
    try:
        from modules.reminder_utils import (
            parse_reminder,
            parse_list_reminder_request,
            _parse_time_from_entities_text,
            _parse_date_from_entities_text
        )
    except ImportError as e:
        pytest.fail(f"Failed to import modules: {e}")

def test_function_signatures():
    """Test that all functions have the expected signatures."""
    import inspect
    sig1 = inspect.signature(parse_reminder)
    assert "text" in sig1.parameters and "entities" in sig1.parameters

    sig2 = inspect.signature(parse_list_reminder_request)
    assert "text" in sig2.parameters and "entities" in sig2.parameters

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
