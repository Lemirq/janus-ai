import os
import json
from datetime import datetime
from typing import Dict, Any


# Settings file path
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
os.makedirs(DATA_DIR, exist_ok=True)
SETTINGS_PATH = os.path.join(DATA_DIR, "settings.json")


# Speed definitions
SPEED_VALUES = ["slow", "medium", "fast"]
SPEED_MULTIPLIERS = {
    "slow": 0.75,
    "medium": 1.0,
    "fast": 1.5,
}


def get_default_settings() -> Dict[str, Any]:
    """Returns the default settings structure."""
    return {
        "playbackSpeed": "medium",
        "verbose": False,
        "lastUpdated": datetime.utcnow().isoformat() + "Z",
    }


def read_settings() -> Dict[str, Any]:
    """
    Reads settings from the settings file.
    Creates default settings if file doesn't exist.
    """
    if not os.path.exists(SETTINGS_PATH):
        default = get_default_settings()
        write_settings(default)
        return default
    
    try:
        with open(SETTINGS_PATH, "r") as f:
            settings = json.load(f)
            # Ensure all required keys exist
            default = get_default_settings()
            for key in default:
                if key not in settings:
                    settings[key] = default[key]
            return settings
    except Exception as e:
        print(f"[settings] Error reading settings file: {e}")
        return get_default_settings()


def write_settings(settings: Dict[str, Any]) -> None:
    """Writes settings to the settings file."""
    settings["lastUpdated"] = datetime.utcnow().isoformat() + "Z"
    with open(SETTINGS_PATH, "w") as f:
        json.dump(settings, f, indent=2)


def get_speed_multiplier(speed: str = None) -> float:
    """
    Returns the multiplier for a given speed setting.
    If speed is None, reads current setting from file.
    """
    if speed is None:
        settings = read_settings()
        speed = settings.get("playbackSpeed", "medium")
    
    return SPEED_MULTIPLIERS.get(speed, 1.0)


def increase_speed() -> Dict[str, Any]:
    """
    Cycles playback speed to the next level: slow → medium → fast → slow.
    Returns updated settings.
    """
    settings = read_settings()
    current_speed = settings.get("playbackSpeed", "medium")
    
    try:
        current_index = SPEED_VALUES.index(current_speed)
        next_index = (current_index + 1) % len(SPEED_VALUES)
        new_speed = SPEED_VALUES[next_index]
    except ValueError:
        # If current speed is invalid, default to medium
        new_speed = "medium"
    
    settings["playbackSpeed"] = new_speed
    write_settings(settings)
    
    if settings.get("verbose"):
        print(f"[settings] Increased speed: {current_speed} → {new_speed} ({get_speed_multiplier(new_speed)}x)")
    
    return settings


def decrease_speed() -> Dict[str, Any]:
    """
    Cycles playback speed to the previous level: fast → medium → slow → fast.
    Returns updated settings.
    """
    settings = read_settings()
    current_speed = settings.get("playbackSpeed", "medium")
    
    try:
        current_index = SPEED_VALUES.index(current_speed)
        prev_index = (current_index - 1) % len(SPEED_VALUES)
        new_speed = SPEED_VALUES[prev_index]
    except ValueError:
        # If current speed is invalid, default to medium
        new_speed = "medium"
    
    settings["playbackSpeed"] = new_speed
    write_settings(settings)
    
    if settings.get("verbose"):
        print(f"[settings] Decreased speed: {current_speed} → {new_speed} ({get_speed_multiplier(new_speed)}x)")
    
    return settings


def toggle_verbose() -> Dict[str, Any]:
    """
    Toggles verbose mode on/off.
    Returns updated settings.
    """
    settings = read_settings()
    current_verbose = settings.get("verbose", False)
    new_verbose = not current_verbose
    
    settings["verbose"] = new_verbose
    write_settings(settings)
    
    print(f"[settings] Verbose mode: {'ON' if new_verbose else 'OFF'}")
    
    return settings

