# calendar_utils.py
# Utilities for calendar event management and iCalendar (.ics) file creation

import datetime
from ics import Calendar, Event
from typing import Optional
import os

CALENDAR_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assistant_calendar.ics')

def add_event_to_calendar(summary: str, start: datetime.datetime, end: Optional[datetime.datetime] = None, description: str = "") -> str:
    """
    Add an event to the assistant's .ics calendar file.
    """
    if end is None:
        end = start + datetime.timedelta(hours=1)
    if os.path.exists(CALENDAR_FILE):
        with open(CALENDAR_FILE, 'r', encoding='utf-8') as f:
            c = Calendar(f.read())
    else:
        c = Calendar()
    e = Event()
    e.name = summary
    e.begin = start
    e.end = end
    e.description = description
    c.events.add(e)
    with open(CALENDAR_FILE, 'w', encoding='utf-8') as f:
        f.writelines(c.serialize_iter())
    return f"Event '{summary}' added to calendar."

def get_calendar_file_path() -> str:
    return CALENDAR_FILE
