import re
from datetime import datetime, timedelta, date
from typing import Optional, Dict, Any # Added for type hints

# Helper function to parse time string from entities
def _parse_time_from_entities_text(time_str: str, date_ref_str: Optional[str] = None) -> Optional[datetime]:
    now = datetime.now()
    parsed_time_obj = None
    base_date = now.date()

    if date_ref_str:
        date_ref_str = date_ref_str.lower()
        if "tomorrow" in date_ref_str:
            base_date = now.date() + timedelta(days=1)
        elif "today" in date_ref_str:
            base_date = now.date()
        # Add more date_reference parsing if needed, e.g., specific days of week, "next week"
        # For now, keeping it simple for "tomorrow" and "today" combined with a time.
        # More complex date_references might be better handled by a full parse if `time_str` is just a time.

    # Try to parse HH:MM am/pm or HH:MM
    time_match = re.search(r"(\d{1,2}:\d{2})\s*(am|pm)?", time_str, re.IGNORECASE)
    if time_match:
        time_digits = time_match.group(1)
        am_pm = time_match.group(2)
        try:
            if am_pm:
                parsed_time_obj = datetime.strptime(time_digits + " " + am_pm, "%I:%M %p").time()
            else:
                parsed_time_obj = datetime.strptime(time_digits, "%H:%M").time()
        except ValueError:
            pass # Could not parse time digits

    if parsed_time_obj:
        reminder_datetime = datetime.combine(base_date, parsed_time_obj)
        # If the time refers to today but is in the past, and no explicit date reference was "today",
        # assume it's for the next day (e.g. "remind me at 7am" when it's 10am -> 7am tomorrow)
        # If date_ref_str was "today", respect it.
        if reminder_datetime < now and not (date_ref_str and "today" in date_ref_str):
            if base_date == now.date(): # only adjust if it was implicitly today
                 reminder_datetime = datetime.combine(now.date() + timedelta(days=1), parsed_time_obj)
        return reminder_datetime

    # Handle simple "tomorrow", "today" (as time_phrase, defaulting to 9 AM)
    # This part is more for when `time_phrase` itself is "tomorrow" rather than a time like "at 3pm"
    if "tomorrow" in time_str.lower() and not date_ref_str: # Avoid double counting if date_ref_str is also tomorrow
        return (now + timedelta(days=1)).replace(hour=9, minute=0, second=0, microsecond=0)
    if "today" in time_str.lower() and not date_ref_str:
        return now.replace(hour=9, minute=0, second=0, microsecond=0) # Default 9 AM today if not past

    # Add more parsing for "in X hours/minutes" if that's expected in entities.get("time_phrase")
    in_duration_match = re.search(r"in\s+(\d+)\s+(hour|hours|minute|minutes)", time_str, re.IGNORECASE)
    if in_duration_match:
        num, unit = int(in_duration_match.group(1)), in_duration_match.group(2).lower()
        if "hour" in unit:
            return now + timedelta(hours=num)
        elif "minute" in unit:
            return now + timedelta(minutes=num)

    return None

# Helper function to parse date string from entities (for list reminders)
def _parse_date_from_entities_text(date_ref_str: str) -> Optional[date]:
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


def parse_reminder(text: str, entities: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    if entities:
        task_entity = entities.get("task")
        time_phrase_entity = entities.get("time_phrase")
        date_reference_entity = entities.get("date_reference")
        time_entity = entities.get("time") # Specific time like "3pm" or "7:00"

        parsed_reminder_time_from_entities = None

        if time_phrase_entity: # "remind me to X tomorrow at 3pm", time_phrase = "tomorrow at 3pm"
            parsed_reminder_time_from_entities = _parse_time_from_entities_text(time_phrase_entity)
        elif date_reference_entity and time_entity: # "remind me to X on tuesday at 7am"
            # Combine date_reference and time before parsing
            combined_time_text = f"{date_reference_entity} at {time_entity}"
            parsed_reminder_time_from_entities = _parse_time_from_entities_text(time_entity, date_reference_entity)
        elif time_entity: # "remind me to X at 7am" (implies today or next suitable day)
             parsed_reminder_time_from_entities = _parse_time_from_entities_text(time_entity)
        elif date_reference_entity: # "remind me to X tomorrow" (implies default time e.g. 9am)
             parsed_reminder_time_from_entities = _parse_time_from_entities_text(date_reference_entity)


        if task_entity and parsed_reminder_time_from_entities:
            return {"task": str(task_entity).strip(), "time": parsed_reminder_time_from_entities}
        # If only task entity is present, could fall through to regex which might pick up time, or return task with no time?
        # For now, if entities don't give both, fall through.

    # Fallback to original regex-based parsing if entities are not sufficient
    task_match = re.search(r"remind me to (.*?)(?=(?:on|at|in|tomorrow|today|next|this|last)\b|$)", text, re.IGNORECASE)
    if not task_match:
        return None
    task = task_match.group(1).strip()

    time_text_part = text[task_match.end():].strip().lower()
    if not time_text_part:
        time_text_part = text.lower() # Fallback if time phrase is not immediately after task

    now = datetime.now() # now needs to be defined for regex part too
    reminder_time = None

    # tomorrow at HH:MM am/pm | at HH:MM am/pm tomorrow
    tomorrow_at_time_match = re.search(r"(tomorrow\s+at\s+(\d{1,2}:\d{2}\s*(?:am|pm)?)|at\s+(\d{1,2}:\d{2}\s*(?:am|pm)?)\s+tomorrow)", time_text_part, re.IGNORECASE)
    if tomorrow_at_time_match:
        time_str = (tomorrow_at_time_match.group(2) or tomorrow_at_time_match.group(3)).strip()
        try:
            parsed_time = datetime.strptime(time_str, "%I:%M %p").time()
        except ValueError:
            try:
                parsed_time = datetime.strptime(time_str, "%H:%M").time()
            except ValueError:
                return None
        reminder_time = datetime.combine(now + timedelta(days=1), parsed_time)

    # at HH:MM am/pm (today or next day if past)
    if not reminder_time:
        at_time_match = re.search(r"at\s+(\d{1,2}:\d{2}\s*(?:am|pm)?)", time_text_part, re.IGNORECASE)
        if at_time_match:
            time_str = at_time_match.group(1).strip()
            try:
                parsed_time = datetime.strptime(time_str, "%I:%M %p").time()
            except ValueError:
                try:
                    parsed_time = datetime.strptime(time_str, "%H:%M").time()
                except ValueError:
                    return None

            rt_today = datetime.combine(now.date(), parsed_time)
            reminder_time = rt_today if rt_today >= now else datetime.combine(now.date() + timedelta(days=1), parsed_time)

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

def parse_list_reminder_request(text: str, entities: Optional[Dict[str, Any]] = None) -> Optional[date]:
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
    text_lower = text.lower() # Ensure text is lowercased for regex part
    now = datetime.now()

    if "today" in text_lower:
        return now.date()
    if "tomorrow" in text_lower:
        return (now + timedelta(days=1)).date()
    if "yesterday" in text_lower:
        return (now - timedelta(days=1)).date()

    rel_pattern = r"in (\d+) (day|days|week|weeks|month|months)"
    match = re.search(rel_pattern, text_lower)
    if match:
        num, unit = int(match.group(1)), match.group(2)
        if "day" in unit:
            return (now + timedelta(days=num)).date()
        if "week" in unit:
            return (now + timedelta(weeks=num)).date()
        if "month" in unit: # Approx
            return (now + timedelta(days=30 * num)).date()

    if "next week" in text_lower:
        days_ahead = (0 - now.weekday() + 7) % 7 or 7
        return (now + timedelta(days=days_ahead)).date()
    if "this week" in text_lower: # Simplified
        return now.date()

    days_of_week = {"monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6}
    for day_name, day_idx in days_of_week.items():
        if day_name in text_lower:
            current_day_idx = now.weekday()
            days_to_add = (day_idx - current_day_idx + 7) % 7
            if "next" in text_lower and days_to_add == 0: # e.g. "next monday" when today is monday
                days_to_add = 7
            if "this" in text_lower and day_idx < current_day_idx : # e.g. "this monday" when today is Wed -> past Mon.
                 days_to_add = day_idx - current_day_idx
            return (now + timedelta(days=days_to_add)).date()

    month_pattern = r"(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{1,2})(?:st|nd|rd|th)?(?:,\s*(\d{4}))?"
    match = re.search(month_pattern, text_lower, re.IGNORECASE)
    if match:
        month_name, day_str, year_str = match.groups()
        try:
            month = datetime.strptime(month_name, "%B").month
            day = int(day_str)
            year = int(year_str) if year_str else now.year
            target_date = date(year, month, day)
            # If the parsed date is in the past, and no explicit year was given, and not "last month" etc.
            if target_date < now.date() and not year_str and not any(kw in text_lower for kw in ["last", "past"]):
                target_date = date(year + 1, month, day) # Assume next year
            return target_date
        except ValueError:
            return None # Malformed date

    date_patterns = [
        r"on (\d{4}-\d{1,2}-\d{1,2})", r"on (\d{1,2}\/\d{1,2}\/\d{4})", # YYYY-MM-DD or MM/DD/YYYY after "on"
        r"(\d{4}-\d{1,2}-\d{1,2})", r"(\d{1,2}\/\d{1,2}\/\d{4})" # Same formats, standalone
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
