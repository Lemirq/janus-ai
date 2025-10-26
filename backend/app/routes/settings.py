from flask import Blueprint, jsonify, request
from ..settings import (
    read_settings,
    write_settings,
    increase_speed,
    decrease_speed,
    toggle_verbose,
    get_speed_multiplier,
    SPEED_VALUES,
)


bp = Blueprint("settings", __name__)


@bp.get("/settings")
def get_settings():
    """
    GET /api/settings
    Returns current settings with speed multiplier.
    """
    settings = read_settings()
    speed = settings.get("playbackSpeed", "medium")
    
    return jsonify({
        **settings,
        "speedMultiplier": get_speed_multiplier(speed),
    })


@bp.post("/settings")
def update_settings():
    """
    POST /api/settings
    Updates settings with provided values.
    
    Body:
    {
      "playbackSpeed": "fast",  // optional: "slow", "medium", or "fast"
      "verbose": true            // optional: true or false
    }
    """
    payload = request.get_json(force=True, silent=True) or {}
    settings = read_settings()
    
    # Update playback speed if provided
    if "playbackSpeed" in payload:
        new_speed = payload["playbackSpeed"]
        if new_speed in SPEED_VALUES:
            settings["playbackSpeed"] = new_speed
        else:
            return jsonify({
                "error": f"invalid playbackSpeed, must be one of: {', '.join(SPEED_VALUES)}"
            }), 400
    
    # Update verbose if provided
    if "verbose" in payload:
        if isinstance(payload["verbose"], bool):
            settings["verbose"] = payload["verbose"]
        else:
            return jsonify({"error": "verbose must be a boolean"}), 400
    
    # Save settings
    write_settings(settings)
    
    speed = settings.get("playbackSpeed", "medium")
    return jsonify({
        **settings,
        "speedMultiplier": get_speed_multiplier(speed),
    })


@bp.post("/settings/speed/increase")
def increase_playback_speed():
    """
    POST /api/settings/speed/increase
    Cycles speed: slow → medium → fast → slow
    """
    settings = increase_speed()
    speed = settings.get("playbackSpeed", "medium")
    
    return jsonify({
        **settings,
        "speedMultiplier": get_speed_multiplier(speed),
    })


@bp.post("/settings/speed/decrease")
def decrease_playback_speed():
    """
    POST /api/settings/speed/decrease
    Cycles speed: fast → medium → slow → fast
    """
    settings = decrease_speed()
    speed = settings.get("playbackSpeed", "medium")
    
    return jsonify({
        **settings,
        "speedMultiplier": get_speed_multiplier(speed),
    })


@bp.post("/settings/verbose")
def toggle_verbose_mode():
    """
    POST /api/settings/verbose
    Toggles verbose mode on/off.
    """
    settings = toggle_verbose()
    
    return jsonify(settings)

