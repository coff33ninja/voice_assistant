import functools
import asyncio
import logging
from typing import Optional

# Centralized async error handler decorator with optional timeout and TTS feedback

def async_error_handler(timeout: Optional[float] = None):
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                if timeout is not None:
                    return await asyncio.wait_for(func(*args, **kwargs), timeout)
                else:
                    return await func(*args, **kwargs)
            except asyncio.TimeoutError:
                logging.error(f"Timeout in {func.__name__} after {timeout} seconds.")
                tts_callback = kwargs.get('tts_callback')
                if tts_callback:
                    await tts_callback(f"Sorry, the operation timed out after {timeout} seconds.")
            except Exception as e:
                logging.exception(f"Error in {func.__name__}: {e}")
                tts_callback = kwargs.get('tts_callback')
                if tts_callback:
                    await tts_callback(f"Sorry, an error occurred in {func.__name__}.")
        return wrapper
    return decorator
