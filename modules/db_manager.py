import sqlite3
import asyncio
from datetime import datetime, date
from .config import DB_PATH

def initialize_db():
    print(f"Initializing database at {DB_PATH}...")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS reminders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task TEXT NOT NULL,
            reminder_time TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()
    print("Database initialized.")

async def save_reminder_async(task: str, reminder_time: datetime):
    # This function can be called directly with await asyncio.to_thread
    # or be fully async with an async DB library if performance becomes an issue.
    def _save():
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO reminders (task, reminder_time, created_at) VALUES (?, ?, ?)",
            (task, reminder_time.isoformat(), datetime.now().isoformat()),
        )
        conn.commit()
        conn.close()
    await asyncio.to_thread(_save)

async def get_reminders_for_date_async(target_date: date) -> list[dict]:
    def _fetch():
        reminders_on_date = []
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        try:
            date_str = target_date.isoformat()
            cursor.execute(
                "SELECT task, reminder_time FROM reminders WHERE date(reminder_time) = ?",
                (date_str,),
            )
            for task, reminder_time_str in cursor.fetchall():
                reminders_on_date.append({"task": task, "time": datetime.fromisoformat(reminder_time_str)})
        except Exception as e:
            print(f"Error fetching reminders for date {target_date}: {e}")
        finally:
            conn.close()
        return reminders_on_date
    return await asyncio.to_thread(_fetch)

async def reminder_check_loop(tts_service_speak_async_callback):
    print("Reminder check loop started.")
    while True:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        current_time_iso = datetime.now().isoformat()
        cursor.execute(
            "SELECT id, task FROM reminders WHERE reminder_time <= ?", (current_time_iso,)
        )
        due_reminders = cursor.fetchall()
        for r_id, task in due_reminders:
            print(f"Reminder: {task}")
            await tts_service_speak_async_callback(f"Reminder: {task}")
            cursor.execute("DELETE FROM reminders WHERE id = ?", (r_id,))
        conn.commit()
        conn.close()
        await asyncio.sleep(60) # Check every 60 seconds