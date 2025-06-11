"""
Module: system_info.py
Provides system information such as CPU, memory, and disk usage via TTS.
"""

import psutil
import os
import platform
import datetime
import time # For boot_time
import logging
from core.tts import speak
from typing import Optional

def bytes_to_gb(bytes_val: int) -> str:
    """
    Converts a byte value to a string representing gigabytes with two decimal places.
    
    Args:
        bytes_val: The value in bytes to convert.
    
    Returns:
        A string representing the equivalent value in gigabytes, formatted to two decimal places.
    """
    gb_val = bytes_val / (1024**3)
    return f"{gb_val:.2f} GB"

def format_uptime(seconds: int) -> str:
    """
    Converts a duration in seconds into a human-readable string with days, hours, and minutes.
    
    Args:
        seconds: The uptime duration in seconds.
    
    Returns:
        A string expressing the duration in days, hours, and minutes with proper pluralization and conjunctions. Returns "less than a minute" if the duration is under one minute.
    """
    days = int(seconds // (24 * 3600))
    seconds %= (24 * 3600)
    hours = int(seconds // 3600)
    seconds %= 3600
    minutes = int(seconds // 60)
    parts = []
    if days > 0:
        parts.append(f"{days} day{'s' if days != 1 else ''}")
    if hours > 0:
        parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
    if minutes > 0:
        parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
    if not parts:
        return "less than a minute"
    elif len(parts) == 1:
        return parts[0]
    elif len(parts) == 2:
        return " and ".join(parts)
    else:
        return ", ".join(parts[:-1]) + ", and " + parts[-1]

def get_cpu_usage_speak() -> None:
    """
    Retrieves the current CPU usage percentage and announces it via text-to-speech.
    
    If CPU usage cannot be determined or an error occurs, a spoken error message is provided.
    """
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        if cpu_percent is not None:
            speak(f"Current CPU usage is {cpu_percent:.1f} percent.")
            logging.info(f"CPU Usage: {cpu_percent:.1f}%")
        else:
            speak("Sorry, I couldn't determine the current CPU usage.")
            logging.error("psutil.cpu_percent() returned None.")
    except Exception as e:
        logging.error(f"Error getting CPU usage: {e}", exc_info=True)
        speak("Sorry, I encountered an error while trying to get CPU usage.")

def get_memory_usage_speak() -> None:
    """
    Retrieves current memory usage statistics and announces them via text-to-speech.
    
    Speaks the percentage of memory used, along with total and used memory in gigabytes. Logs detailed memory information. If retrieval fails, announces an error message.
    """
    try:
        mem = psutil.virtual_memory()
        total_mem = bytes_to_gb(mem.total)
        used_mem = bytes_to_gb(mem.used)
        mem_percent = mem.percent
        speak(f"Memory usage is {mem_percent:.1f} percent. Total: {total_mem}, Used: {used_mem}.")
        logging.info(f"Memory Usage: {mem_percent:.1f}% (Total: {total_mem}, Used: {used_mem}, Available: {bytes_to_gb(mem.available)})")
    except Exception as e:
        logging.error(f"Error getting memory usage: {e}", exc_info=True)
        speak("Sorry, I encountered an error while trying to get memory usage.")

def get_disk_usage_speak(path_argument: Optional[str] = None) -> None:
    """
    Speaks the disk usage statistics for a specified path or the main drive.
    
    If no path is provided, checks the main drive (C:\ on Windows, / on Unix-like systems). Announces the percentage of disk space used, total disk size, and free space. If the path does not exist or an error occurs, informs the user accordingly.
    """
    target_path = path_argument
    path_display_name = "the main drive"
    if not target_path:
        if platform.system().lower() == "windows":
            target_path = r"C:\\"
            path_display_name = "drive C"
        else:
            target_path = "/" # Root for Linux/macOS
            path_display_name = "the root filesystem"
        logging.info(f"No path specified for disk usage, defaulting to '{target_path}'.")
    else:
        path_display_name = f"the path {target_path}"
        logging.info(f"Checking disk usage for specified path: '{target_path}'")
    try:
        if not os.path.exists(target_path):
            speak(f"Sorry, I could not find the path {path_display_name} to check its disk space.")
            logging.error(f"Path '{target_path}' for disk usage check does not exist.")
            return
        disk = psutil.disk_usage(target_path)
        total_disk = bytes_to_gb(disk.total)
        free_disk = bytes_to_gb(disk.free)
        disk_percent = disk.percent
        speak(f"Disk usage for {path_display_name}: {disk_percent:.1f} percent used. Total: {total_disk}, Free: {free_disk}.")
        logging.info(f"Disk Usage for {target_path}: {disk_percent:.1f}% (Total: {total_disk}, Free: {free_disk})")
    except Exception as e:
        logging.error(f"Error getting disk usage for {target_path}: {e}", exc_info=True)
        speak(f"Sorry, I encountered an error while trying to get disk usage for {path_display_name}.")

def get_system_uptime_speak():
    """
    Speaks the system uptime as a human-readable duration.
    
    Calculates the time elapsed since the system booted and announces it using text-to-speech. Logs the uptime and boot timestamp. If an error occurs, a spoken error message is provided.
    """
    try:
        boot_timestamp = psutil.boot_time()
        current_timestamp = time.time()
        uptime_seconds = current_timestamp - boot_timestamp
        uptime_str = format_uptime(int(uptime_seconds))
        speak(f"The system has been running for {uptime_str}.")
        logging.info(f"System Uptime: {uptime_str} (since {datetime.datetime.fromtimestamp(boot_timestamp).strftime('%Y-%m-%d %H:%M:%S')})")
    except Exception as e:
        logging.error(f"Error getting system uptime: {e}", exc_info=True)
        speak("Sorry, I encountered an error while trying to get system uptime.")

def get_system_summary_speak():
    """
    Speaks a summary of the system's CPU usage, memory usage, main disk usage, and uptime.
    
    Attempts to retrieve each metric individually, providing a combined spoken summary of available information. If no data can be retrieved, informs the user accordingly.
    """
    speak("Getting system status summary.")
    logging.info("ACTION: Getting system status summary...")

    summary_parts = []
    try:
        cpu_percent = psutil.cpu_percent(interval=0.5) # Shorter interval for summary
        if cpu_percent is not None:
            summary_parts.append(f"CPU at {cpu_percent:.1f} percent")
    except Exception:
        pass # Ignore individual errors for summary, just skip part

    try:
        mem = psutil.virtual_memory()
        summary_parts.append(f"memory at {mem.percent:.1f} percent")
    except Exception:
        pass

    try:
        default_disk_path = r"C:\\" if platform.system().lower() == "windows" else "/"
        if os.path.exists(default_disk_path): # Check existence before calling psutil.disk_usage
             disk = psutil.disk_usage(default_disk_path)
             summary_parts.append(f"main disk at {disk.percent:.1f} percent")
    except Exception:
        pass

    try:
        boot_timestamp = psutil.boot_time()
        uptime_seconds = time.time() - boot_timestamp
        uptime_str = format_uptime(int(uptime_seconds))
        summary_parts.append(f"uptime is {uptime_str}")
    except Exception:
        pass

    if summary_parts:
        speak("System status: " + ", ".join(summary_parts) + ".")
        logging.info("System Summary: %s.", ", ".join(summary_parts))
    else:
        speak("Sorry, I couldn't retrieve any system status information.")
        logging.error("Failed to retrieve any parts for system summary.")


def get_cpu_load_speak(): # Renamed from get_cpu_load to avoid conflict if old one existed and for consistency
    """
    Retrieves and speaks the system load averages for 1, 5, and 15 minutes.
    
    If load averages are unavailable on the current operating system, informs the user accordingly.
    """
    if hasattr(psutil, "getloadavg"): # Check if function exists (not on Windows)
        try:
            load_avg = psutil.getloadavg() # Returns a tuple (1min, 5min, 15min)
            response = (f"System load averages are: "
                        f"{load_avg[0]:.2f} for one minute, "
                        f"{load_avg[1]:.2f} for five minutes, "
                        f"and {load_avg[2]:.2f} for fifteen minutes.")
            logging.info(f"ACTION: {response}")
            speak(response)
        except Exception as e:
            logging.error(f"Error getting CPU load average: {e}", exc_info=True)
            speak("Sorry, I encountered an error trying to get the system load average.")
    else:
        msg = "System load average is not available on this operating system."
        logging.info(f"INFO: {msg}")
        speak(msg)


def register_intents():
    """
    Returns a mapping of voice command intents to their corresponding system information functions.
    
    The returned dictionary associates various user phrases with functions that provide spoken system status, CPU usage, memory usage, disk usage, uptime, and load averages. This enables integration with a voice assistant framework for handling system information queries.
    """
    intents = {
        "system status": get_system_summary_speak,
        "tell me system status": get_system_summary_speak,
        "what's the system status": get_system_summary_speak,

        "cpu usage": get_cpu_usage_speak,
        "what's the cpu usage": get_cpu_usage_speak,
        "tell me cpu load": get_cpu_load_speak, # Points to the new get_cpu_load_speak

        "memory usage": get_memory_usage_speak,
        "what's the memory usage": get_memory_usage_speak,
        "ram status": get_memory_usage_speak,

        # For "disk space", main.py's argument parser will pass anything after "disk space "
        # or "disk space for " as the 'path_argument' to get_disk_usage_speak.
        # If nothing is passed (e.g. user just says "disk space"), path_argument will be None or empty.
        "disk space": get_disk_usage_speak,
        "disk space for": get_disk_usage_speak, # "disk space for /home" -> path_argument="/home"
        "how much disk space is left": get_disk_usage_speak, # Will use default path
        "storage status": get_disk_usage_speak, # Will use default path

        "system uptime": get_system_uptime_speak,
        "how long has the system been running": get_system_uptime_speak,

        "system load": get_cpu_load_speak,
        "what's the system load": get_cpu_load_speak,
        "what is the system load average": get_cpu_load_speak
    }
    return intents
