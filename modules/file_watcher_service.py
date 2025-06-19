# file_watcher_service.py
import time
import logging
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from typing import Callable, List

logger = logging.getLogger(__name__)

class NormalizationDataChangeHandler(FileSystemEventHandler):
    """Handles file system events for normalization data files."""
    def __init__(self, reload_callback: Callable[[], None], files_to_watch: List[str]):
        super().__init__()
        self.reload_callback = reload_callback
        # Store absolute paths for reliable comparison
        self.files_to_watch = [os.path.abspath(f) for f in files_to_watch]
        self.last_triggered_time = {} # To debounce events
        self.debounce_interval = 2 # seconds

    def on_any_event(self, event):
        """
        Catches all events (created, deleted, modified, moved)
        and triggers reload if a watched file is affected.
        """
        if event.is_directory:
            return

        event_path = os.path.abspath(event.src_path)
        
        # For 'moved' events, check the destination path as well
        dest_path = os.path.abspath(getattr(event, 'dest_path', event.src_path))

        if event_path in self.files_to_watch or dest_path in self.files_to_watch:
            current_time = time.time()
            # Debounce: if this file was triggered recently, ignore
            if event_path in self.last_triggered_time and \
               current_time - self.last_triggered_time[event_path] < self.debounce_interval:
                return
            
            self.last_triggered_time[event_path] = current_time
            logger.info(f"Detected change in '{event.src_path}'. Triggering normalization data reload.")
            try:
                self.reload_callback()
            except Exception as e:
                logger.error(f"Error during normalization data reload callback: {e}", exc_info=True)

def start_normalization_data_watcher(reload_callback: Callable[[], None], data_dir: str, filenames: List[str]):
    """
    Starts a file system watcher for the normalization data files.
    This function should be run in a separate thread.
    """
    files_to_watch = [os.path.join(data_dir, fname) for fname in filenames]
    event_handler = NormalizationDataChangeHandler(reload_callback, files_to_watch)
    observer = Observer()
    observer.schedule(event_handler, data_dir, recursive=False) # Watch only the specified directory
    observer.start()
    logger.info(f"File watcher started for normalization data in '{data_dir}'. Watching: {filenames}")
    try:
        while observer.is_alive(): # Keep the thread alive while observer is running
            observer.join(1)
    except KeyboardInterrupt: # Allow graceful shutdown
        logger.info("File watcher received KeyboardInterrupt.")
    finally:
        observer.stop()
        observer.join()
        logger.info("File watcher stopped.")