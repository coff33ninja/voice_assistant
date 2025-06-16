import tkinter as tk
from datetime import datetime

def show_reminders_gui(reminders: list[dict], date_str: str):
    """
    Display reminders in a simple Tkinter window.
    This function is blocking and should be run in a separate thread.
    """
    root = tk.Tk()
    root.title(f"Reminders for {date_str}")
    
    text_widget = tk.Text(root, wrap=tk.WORD, font=("Arial", 12), padx=10, pady=10, bg="#f0f0f0", relief=tk.FLAT)
    text_widget.insert(tk.END, f"Reminders for {date_str}:\n\n")

    if reminders:
        for r_item in reminders:
            task = r_item.get('task', 'No task description')
            time_val = r_item.get('time')
            if isinstance(time_val, datetime):
                time_str_display = time_val.strftime('%I:%M %p')
            elif isinstance(time_val, str): # If already stringified
                try: time_str_display = datetime.fromisoformat(time_val).strftime('%I:%M %p')
                except ValueError: time_str_display = time_val # show as is
            else:
                time_str_display = "Unknown time"
            text_widget.insert(tk.END, f"- {task} at {time_str_display}\n")
    else:
        text_widget.insert(tk.END, "No reminders scheduled.")
    
    text_widget.config(state=tk.DISABLED)
    text_widget.pack(expand=True, fill=tk.BOTH)
    
    btn = tk.Button(root, text="Close", command=root.destroy, font=("Arial", 12), width=10)
    btn.pack(pady=10)
    root.mainloop()