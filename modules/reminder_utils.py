import re
from datetime import datetime, timedelta, date  # Removed unused time and datetime imports
import logging # Import logging
from typing import Optional, Dict, Any  # Removed unused Union

# Helper function to parse time string from entities
# Added optional 'now' parameter for testability

def _parse_time_from_entities_text(time_str: Optional[str], date_ref_str: Optional[str] = None, now: Optional[datetime] = None) -> Optional[datetime]:
    if not time_str:
        return None

    if now is None:
        now = datetime.now()
    parsed_time_obj = None
    base_date = now.date()

    # Handle relative time first: "in X hours/minutes"
    in_duration_match = re.fullmatch(r"in\s+(\d+)\s+(hour|hours|minute|minutes)", time_str.strip(), re.IGNORECASE)
    if in_duration_match:
        num, unit = int(in_duration_match.group(1)), in_duration_match.group(2).lower()
        if "hour" in unit:
            return now + timedelta(hours=num)
        elif "minute" in unit:
            return now + timedelta(minutes=num)

    if date_ref_str:
        date_ref_str = date_ref_str.lower()
        if "tomorrow" in date_ref_str:
            base_date = now.date() + timedelta(days=1)
        elif "today" in date_ref_str:
            base_date = now.date()

    if not date_ref_str:
        time_str_lower = time_str.lower().strip()
        # Only strip 'tomorrow' or 'today' if the string is not exactly 'tomorrow' or 'today'
        if time_str_lower != "tomorrow" and "tomorrow" in time_str_lower:
            time_str = re.sub(r'tomorrow\s*(at\s*)?', '', time_str, flags=re.IGNORECASE).strip()
            base_date = now.date() + timedelta(days=1)
        elif time_str_lower != "today" and "today" in time_str_lower:
            time_str = re.sub(r'today\s*(at\s*)?', '', time_str, flags=re.IGNORECASE).strip()
            base_date = now.date()

    # Try to parse time like "3pm", "3:30pm", "15:00" from the (potentially cleaned) time string
    time_pattern_match = re.fullmatch(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", time_str.strip(), re.IGNORECASE)
    if time_pattern_match:
        hour_str = time_pattern_match.group(1)
        minute_str = time_pattern_match.group(2)
        am_pm_str = time_pattern_match.group(3)
        time_digits_for_strptime = f"{hour_str}:{minute_str if minute_str is not None else '00'}"
        try:
            if am_pm_str:
                parsed_time_obj = datetime.strptime(time_digits_for_strptime + " " + am_pm_str, "%I:%M %p").time()
            else:
                parsed_time_obj = datetime.strptime(time_digits_for_strptime, "%H:%M").time()
        except ValueError:
            logger = logging.getLogger(__name__)
            logger.debug(f"Could not parse time digits: {time_digits_for_strptime} with/without am/pm: {am_pm_str}")
            pass

    if parsed_time_obj:
        try:
            combined_dt = datetime.combine(base_date, parsed_time_obj)
        except TypeError:
            logger = logging.getLogger(__name__)
            logger.debug(f"Failed to combine base_date '{base_date}' with parsed_time_obj '{parsed_time_obj}' of type {type(parsed_time_obj)}")
            return None
        if combined_dt < now:
            if base_date == now.date() and not (date_ref_str and "today" in date_ref_str.lower()):
                return datetime.combine(now.date() + timedelta(days=1), parsed_time_obj)
        return combined_dt

    cleaned_time_str = time_str.strip().lower()
    if not parsed_time_obj:
        # Accept 'tomorrow' or 'today' with extra spaces
        if cleaned_time_str in ("tomorrow", "today") and not date_ref_str:
            if cleaned_time_str == "tomorrow":
                return (now + timedelta(days=1)).replace(hour=9, minute=0, second=0, microsecond=0)
            else:
                return now.replace(hour=9, minute=0, second=0, microsecond=0)

    return None

# Helper function to parse date string from entities (for list reminders)
def _parse_date_from_entities_text(date_ref_str: Optional[str]) -> Optional[date]:
    if not date_ref_str: # Add check for None or empty string
        return None

    now = datetime.now()
    date_ref_str = date_ref_str.lower()

    if "today" in date_ref_str:
        return now.date()
    if "tomorrow" in date_ref_str:
        return (now + timedelta(days=1)).date()
    if "yesterday" in date_ref_str: # Though less common for list *future* reminders
        return (now - timedelta(days=1)).date()

    # Basic "next week", "this week"
    if "next week" in date_ref_str:
        days_ahead = (0 - now.weekday() + 7) % 7 or 7 # 0 is Monday
        return (now + timedelta(days=days_ahead)).date()
    if "this week" in date_ref_str: # Could mean today or any day this week. For simplicity, use today.
        return now.date() # Or interpret based on other keywords if available

    # Specific days of the week e.g. "next monday", "on tuesday"
    days_of_week = {"monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6}
    for day_name, day_idx in days_of_week.items():
        if day_name in date_ref_str:
            current_day_idx = now.weekday()
            days_to_add = (day_idx - current_day_idx + 7) % 7
            if "next" in date_ref_str and days_to_add == 0: # "next monday" when today is monday
                days_to_add = 7
            # if "this" in text and day_idx < current_day_idx : # e.g. "this monday" when today is wed
            #      days_to_add = day_idx - current_day_idx # results in a past date
            return (now + timedelta(days=days_to_add)).date()

    # Add more specific date parsing (e.g., "July 4th") if necessary
    # This would be similar to the regexes at the end of the original parse_list_reminder_request
    return None

logger = logging.getLogger(__name__) # Add logger

def parse_reminder(text: Optional[str], entities: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    if not text: # Handles None or empty string
        logger.debug(f"parse_reminder called with invalid or no text ({text!r}), returning None.")
        return None

    # Ensure text is not empty for regex operations after the initial None/empty check
    if not isinstance(text, str) or not text.strip(): # text might be "   " or non-string
        logger.debug("parse_reminder called with whitespace-only text, returning None.")
        return None

    if entities:
        task_entity = entities.get("task")
        time_phrase_entity = entities.get("time_phrase")
        # If time_phrase contains a date reference AND a time, _parse_time_from_entities_text handles it.
        # If time_phrase is just a date reference ("tomorrow"), _parse_time_from_entities_text handles it (defaults to 9am).
        # If time_phrase is just a time ("3pm"), _parse_time_from_entities_text handles it (defaults to today/tomorrow).

        date_reference_entity = entities.get("date_reference")
        time_entity = entities.get("time") # Specific time like "3pm" or "7:00"

        parsed_reminder_time_from_entities = None

        if time_phrase_entity: # "remind me to X tomorrow at 3pm", time_phrase = "tomorrow at 3pm"
            parsed_reminder_time_from_entities = _parse_time_from_entities_text(time_phrase_entity)
        elif date_reference_entity and time_entity: # "remind me to X on tuesday at 7am"
            # Combine date_reference and time before parsing
            # Pass date_reference_entity as the date_ref_str argument to _parse_time_from_entities_text
            parsed_reminder_time_from_entities = _parse_time_from_entities_text(
                time_entity, date_reference_entity
            )
        elif time_entity: # "remind me to X at 7am" (implies today or next suitable day)
            parsed_reminder_time_from_entities = _parse_time_from_entities_text(time_entity)
        # Note: If only date_reference_entity is present, _parse_time_from_entities_text("tomorrow") will handle it.
        elif date_reference_entity: # "remind me to X tomorrow" (implies default time e.g. 9am)
            parsed_reminder_time_from_entities = _parse_time_from_entities_text(date_reference_entity)

        if task_entity and parsed_reminder_time_from_entities:
            return {"task": str(task_entity).strip(), "time": parsed_reminder_time_from_entities}
        # If only task entity is present, could fall through to regex which might pick up time, or return task with no time?
        # For now, if entities don't give both, fall through.

    # Fallback to regex-based parsing if entities are not sufficient or not present
    # Ensure task extraction regex is robust
    task_match = re.search(r"remind me to (.*?)(?=(?:on|at|in|tomorrow|today|next|this|last)\b|$)", text, re.IGNORECASE)
    if not task_match: # If the primary regex fails, try a simpler one if "remind me to" is present
        if "remind me to" in text.lower():  # Check for the phrase itself
            task_match = re.search(r"remind me to (.*)", text, re.IGNORECASE)
        if not task_match: # If still no match
            logger.debug(f"Could not extract task from reminder: '{text}' using regex.")
            return None
    task = task_match.group(1).strip()

    time_text_part = text[task_match.end():].strip().lower()
    # If time_text_part is empty, the time/date info might be at the beginning or elsewhere.
    # For simplicity, let's just use the full lowercased text for time/date regex matching if time_text_part is empty.
    if not time_text_part:
        time_text_part = text.lower() # Use full text if no specific time part after task

    now = datetime.now()
    reminder_time = None

    # tomorrow at HH:MM am/pm | at HH:MM am/pm tomorrow
    # Regex for time: H(H)(:MM)? (am/pm)?
    time_regex_flexible = r"(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)"
    tomorrow_at_time_match = re.search(rf"(tomorrow\s+at\s+{time_regex_flexible}|at\s+{time_regex_flexible}\s+tomorrow)", time_text_part, re.IGNORECASE)

    if tomorrow_at_time_match:
        time_str = (tomorrow_at_time_match.group(2) or tomorrow_at_time_match.group(3)).strip()
        try:
            # Use _parse_time_from_entities_text to handle "HH:MM am/pm" and "HH:MM" with tomorrow context
            parsed_dt_obj = _parse_time_from_entities_text(time_str, "tomorrow") # Pass "tomorrow" as date_ref
            if parsed_dt_obj: # Check if it's a datetime object
                reminder_time = parsed_dt_obj
        except ValueError:
            logger.warning(f"Could not parse time '{time_str}' with 'tomorrow' context in regex.")
            pass # Fall through

    # at HH:MM am/pm (today or next day if past)
    if not reminder_time:
        at_time_match = re.search(rf"at\s+{time_regex_flexible}", time_text_part, re.IGNORECASE)
        if at_time_match:
            time_str = at_time_match.group(1).strip()
            try:
                parsed_dt_obj = _parse_time_from_entities_text(time_str) # No explicit date_ref, _parse_time_from_entities_text handles today/next day logic
                if parsed_dt_obj: # Check if it's a datetime object
                    reminder_time = parsed_dt_obj
            except ValueError:
                logger.warning(f"Could not parse time '{time_str}' in regex.")
                pass # Fall through

    # "tomorrow" (defaults to 9 AM)
    if not reminder_time and "tomorrow" in time_text_part: # This regex part can still be a fallback
        reminder_time = (now + timedelta(days=1)).replace(hour=9, minute=0, second=0, microsecond=0)

    # "in X unit" # This regex part can still be a fallback
    if not reminder_time:
        in_duration_match = re.search(r"in\s+(\d+)\s+(hour|hours|minute|minutes|day|days|week|weeks)", time_text_part, re.IGNORECASE)
        if in_duration_match:
            num, unit = int(in_duration_match.group(1)), in_duration_match.group(2).lower()
            if "hour" in unit:
                reminder_time = now + timedelta(hours=num)
            elif "minute" in unit:
                reminder_time = now + timedelta(minutes=num)
            elif "day" in unit:
                reminder_time = now + timedelta(days=num)
            elif "week" in unit:
                reminder_time = now + timedelta(weeks=num)

    if task and reminder_time: # This is from regex parsing
        return {"task": task, "time": reminder_time}

    # If task was found by regex but no time, and we had a task_entity but no time_entity earlier,
    # we might want to return the task_entity with no specific time.
    # However, the current function structure returns None if time is not found by regex.
    # For now, maintaining existing behavior for regex fallback.
    return None

def parse_list_reminder_request(text: Optional[str], entities: Optional[Dict[str, Any]] = None) -> Optional[date]:
    if not text: # Handles None or empty string
        logger.debug("parse_list_reminder_request called with no text, returning None.")
        return None
    text_lower = text.lower()  # Ensure text is lowercased for all regex/keyword checks

    if entities:
        date_reference_entity = entities.get("date_reference")
        if date_reference_entity:
            parsed_date = _parse_date_from_entities_text(str(date_reference_entity))
            if parsed_date:
                return parsed_date
        # If "time_phrase" contains something like "next week" or "July 4th",
        # _parse_date_from_entities_text could be called with entities.get("time_phrase") too.
        # For now, focusing on date_reference.

    # Fallback to original regex-based parsing

    now = datetime.now()

    if (
        "today" in text_lower or "show reminders for today" == text_lower.strip()
    ):  # Handle exact match too
        return now.date()
    if "tomorrow" in text_lower:
        return (now + timedelta(days=1)).date()
    if "yesterday" in text_lower:
        return (now - timedelta(days=1)).date()

    rel_pattern = r"in (\d+) (day|days|week|weeks|month|months)"
    match = re.search(rel_pattern, text_lower)  # Search in the full lowercased text
    if match:
        num, unit = int(match.group(1)), match.group(2)
        if "day" in unit:
            return (now + timedelta(days=num)).date()
        if "week" in unit:
            return (now + timedelta(weeks=num)).date()
        if "month" in unit: # Approx
            return (now + timedelta(days=30 * num)).date()

    if "next week" in text_lower:  # Search in the full lowercased text
        days_ahead = (0 - now.weekday() + 7) % 7 or 7
        return (now + timedelta(days=days_ahead)).date()
    if "this week" in text_lower:  # Search in the full lowercased text
        return now.date()  # Simplified - could mean today or any day this week.

    days_of_week = {"monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6}
    for day_name, day_idx in days_of_week.items():
        if day_name in text_lower:
            current_day_idx = now.weekday()
            days_to_add = (day_idx - current_day_idx + 7) % 7
            if "next" in text_lower and days_to_add == 0: # e.g. "next monday" when today is monday
                days_to_add = 7
            if "this" in text_lower and day_idx < current_day_idx : # e.g. "this monday" when today is Wed -> past Mon.
                days_to_add = day_idx - current_day_idx
            return now.date() + timedelta(days=days_to_add)  # Return date object

    month_pattern = r"(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{1,2})(?:st|nd|rd|th)?(?:,\s*(\d{4}))?"
    match = re.search(month_pattern, text_lower, re.IGNORECASE)
    if match:  # Search in the full lowercased text
        month_name, day_str, year_str = match.groups()
        try:
            month = datetime.strptime(month_name, "%B").month
            day = int(day_str)
            year = int(year_str) if year_str else now.year
            target_date = date(year, month, day)
            # If the parsed date is in the past, and no explicit year was given, and not "last month" etc.
            if target_date < now.date() and not year_str and not any(kw in text_lower for kw in ["last", "past", "yesterday"]):
                target_date = date(year + 1, month, day) # Assume next year
            return target_date
        except ValueError:
            return None # Malformed date

    date_patterns = [
        r"on (\d{4}-\d{1,2}-\d{1,2})",
        r"on (\d{1,2}\/\d{1,2}\/\d{4})",  # YYYY-MM-DD or MM/DD/YYYY after "on"
        r"(\d{4}-\d{1,2}-\d{1,2})",
        r"(\d{1,2}\/\d{1,2}\/\d{4})",  # Same formats, standalone (search in full text)
    ]
    for pat in date_patterns:
        match = re.search(pat, text_lower)
        if match:
            date_str_match = match.group(1) # The date string itself
            try:
                if "-" in date_str_match:
                    return datetime.strptime(date_str_match, "%Y-%m-%d").date()
                if "/" in date_str_match: # Assuming MM/DD/YYYY for "/"
                    return datetime.strptime(date_str_match, "%m/%d/%Y").date()
            except ValueError:
                continue # Try next pattern
    return None
