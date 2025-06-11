import os
from modules import system_info

def test_bytes_to_gb():
    assert system_info.bytes_to_gb(0) == "0.00 GB"
    assert system_info.bytes_to_gb(1024**3) == "1.00 GB"
    assert system_info.bytes_to_gb(2 * 1024**3) == "2.00 GB"

def test_format_uptime():
    assert system_info.format_uptime(0) == "less than a minute"
    assert system_info.format_uptime(60) == "1 minute"
    assert system_info.format_uptime(3600) == "1 hour"
    assert system_info.format_uptime(86400) == "1 day"
    assert system_info.format_uptime(90061) == "1 day, 1 hour, and 1 minute"

# The following tests are smoke tests to ensure no exceptions are raised.
def test_get_cpu_usage_speak(monkeypatch):
    monkeypatch.setattr(system_info, "speak", lambda msg: None)
    system_info.get_cpu_usage_speak()

def test_get_memory_usage_speak(monkeypatch):
    monkeypatch.setattr(system_info, "speak", lambda msg: None)
    system_info.get_memory_usage_speak()

def test_get_disk_usage_speak(monkeypatch):
    monkeypatch.setattr(system_info, "speak", lambda msg: None)
    # Test with default path
    system_info.get_disk_usage_speak()
    # Test with a valid path
    system_info.get_disk_usage_speak(os.getcwd())
    # Test with an invalid path
    system_info.get_disk_usage_speak("/invalid/path/for/test")

def test_get_system_uptime_speak(monkeypatch):
    monkeypatch.setattr(system_info, "speak", lambda msg: None)
    system_info.get_system_uptime_speak()

def test_get_system_summary_speak(monkeypatch):
    monkeypatch.setattr(system_info, "speak", lambda msg: None)
    system_info.get_system_summary_speak()

def test_get_cpu_load_speak(monkeypatch):
    monkeypatch.setattr(system_info, "speak", lambda msg: None)
    system_info.get_cpu_load_speak()

def test_register_intents():
    intents = system_info.register_intents()
    assert isinstance(intents, dict)
    assert "system status" in intents
    assert callable(intents["system status"])
