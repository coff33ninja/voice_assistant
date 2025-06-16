import re
from datetime import datetime, timedelta, date

def parse_reminder(text: str) -> dict | None:
    task_match = re.search(r"remind me to (.*?)(?=(?:on|at|in|tomorrow|today|next|this|last)\b|$)", text, re.IGNORECASE)
    if not task_match:
        return None
    task = task_match.group(1).strip()
    
    time_text_part = text[task_match.end():].strip().lower()
    if not time_text_part:
        time_text_part = text.lower() # Fallback if time phrase is not immediately after task

    now = datetime.now()
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
    if not reminder_time and "tomorrow" in time_text_part:
        reminder_time = (now + timedelta(days=1)).replace(hour=9, minute=0, second=0, microsecond=0)

    # "in X unit"
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

    if task and reminder_time:
        return {"task": task, "time": reminder_time}
    return None

def parse_list_reminder_request(text: str) -> date | None:
    text = text.lower()
    now = datetime.now()

    if "today" in text:
        return now.date()
    if "tomorrow" in text:
        return (now + timedelta(days=1)).date()
    if "yesterday" in text:
        return (now - timedelta(days=1)).date()

    rel_pattern = r"in (\d+) (day|days|week|weeks|month|months)"
    match = re.search(rel_pattern, text)
    if match:
        num, unit = int(match.group(1)), match.group(2)
        if "day" in unit:
            return (now + timedelta(days=num)).date()
        if "week" in unit:
            return (now + timedelta(weeks=num)).date()
        if "month" in unit:
            return (now + timedelta(days=30 * num)).date() # Approx

    if "next week" in text:
        days_ahead = (0 - now.weekday() + 7) % 7 or 7
        return (now + timedelta(days=days_ahead)).date()
    if "this week" in text:
        return now.date()

    days_of_week = {"monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6}
    for day_name, day_idx in days_of_week.items():
        if day_name in text:
            current_day_idx = now.weekday()
            days_to_add = (day_idx - current_day_idx + 7) % 7
            if "next" in text and days_to_add == 0:
                days_to_add = 7
            # Simplified: "this Monday" when today is Wed -> past Mon. "Monday" -> upcoming Mon.
            if "this" in text and day_idx < current_day_idx :
                 days_to_add = day_idx - current_day_idx # past day in current week
            return (now + timedelta(days=days_to_add)).date()

    month_pattern = r"(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{1,2})(?:st|nd|rd|th)?(?:,\s*(\d{4}))?"
    match = re.search(month_pattern, text, re.IGNORECASE)
    if match:
        month_name, day_str, year_str = match.groups()
        try:
            month = datetime.strptime(month_name, "%B").month
            day = int(day_str)
            year = int(year_str) if year_str else now.year
            target_date = date(year, month, day)
            if target_date < now.date() and not year_str and not any(kw in text for kw in ["last", "past"]):
                target_date = date(year + 1, month, day)
            return target_date
        except ValueError:
            return None

    date_patterns = [
        r"on (\d{4}-\d{1,2}-\d{1,2})", r"on (\d{1,2}\/\d{1,2}\/\d{4})",
        r"(\d{4}-\d{1,2}-\d{1,2})", r"(\d{1,2}\/\d{1,2}\/\d{4})"
    ]
    for pat in date_patterns:
        match = re.search(pat, text)
        if match:
            date_str_match = match.group(1)
            try:
                if "-" in date_str_match:
                    return datetime.strptime(date_str_match, "%Y-%m-%d").date()
                if "/" in date_str_match:
                    return datetime.strptime(date_str_match, "%m/%d/%Y").date()
            except ValueError:
                continue
    return None