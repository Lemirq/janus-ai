import os
import json
import uuid
from datetime import datetime
from flask import Blueprint, request, jsonify, send_file


bp = Blueprint("sessions", __name__)


DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "data", "sessions"))
os.makedirs(DATA_DIR, exist_ok=True)
INDEX_PATH = os.path.join(DATA_DIR, "index.json")

# In-memory tracking of last audio clip per session
_session_last_audio = {}


def _read_index():
    if not os.path.exists(INDEX_PATH):
        return {"sessions": []}
    with open(INDEX_PATH, "r") as f:
        try:
            return json.load(f)
        except Exception:
            return {"sessions": []}


def _write_index(data):
    with open(INDEX_PATH, "w") as f:
        json.dump(data, f)


def _session_path(session_id: str) -> str:
    return os.path.join(DATA_DIR, f"{session_id}.json")


@bp.get("/sessions")
def list_sessions():
    return jsonify(_read_index())


@bp.post("/sessions")
def create_session():
    payload = request.get_json(force=True, silent=True) or {}
    objective = (payload.get("objective") or "").strip()
    file_ids = payload.get("fileIds") or []
    collection = payload.get("collection") or "docs"

    if not objective:
        return jsonify({"error": "objective required"}), 400

    session_id = f"sess_{uuid.uuid4().hex[:8]}"
    now = datetime.utcnow().isoformat() + "Z"
    sess = {
        "id": session_id,
        "objective": objective,
        "fileIds": file_ids,
        "collection": collection,
        "createdAt": now,
        "status": "created",
    }

    # Ensure the session file's parent directory exists
    os.makedirs(os.path.dirname(_session_path(session_id)), exist_ok=True)

    # persist session file
    with open(_session_path(session_id), "w") as f:
        json.dump(sess, f)

    # update index
    idx = _read_index()
    idx.setdefault("sessions", []).insert(0, {"id": session_id, "objective": objective, "createdAt": now, "status": "created"})
    _write_index(idx)

    return jsonify(sess)


@bp.post("/sessions/<session_id>/start")
def start_session(session_id: str):
    p = _session_path(session_id)
    if not os.path.exists(p):
        return jsonify({"error": "not found"}), 404
    with open(p, "r") as f:
        sess = json.load(f)
    sess["status"] = "running"
    sess["startedAt"] = datetime.utcnow().isoformat() + "Z"
    with open(p, "w") as f:
        json.dump(sess, f)
    # update index
    idx = _read_index()
    for s in idx.get("sessions", []):
        if s.get("id") == session_id:
            s["status"] = "running"
            break
    _write_index(idx)
    return jsonify({"ok": True})


@bp.post("/sessions/<session_id>/stop")
def stop_session(session_id: str):
    p = _session_path(session_id)
    if not os.path.exists(p):
        return jsonify({"error": "not found"}), 404
    with open(p, "r") as f:
        sess = json.load(f)
    sess["status"] = "stopped"
    sess["stoppedAt"] = datetime.utcnow().isoformat() + "Z"
    with open(p, "w") as f:
        json.dump(sess, f)
    # update index
    idx = _read_index()
    for s in idx.get("sessions", []):
        if s.get("id") == session_id:
            s["status"] = "stopped"
            break
    _write_index(idx)
    return jsonify({"ok": True})



@bp.post("/sessions/<session_id>/complete")
def complete_session(session_id: str):
    p = _session_path(session_id)
    if not os.path.exists(p):
        return jsonify({"error": "not found"}), 404
    with open(p, "r") as f:
        sess = json.load(f)
    sess["status"] = "completed"
    sess["completedAt"] = datetime.utcnow().isoformat() + "Z"
    with open(p, "w") as f:
        json.dump(sess, f)
    # update index
    idx = _read_index()
    for s in idx.get("sessions", []):
        if s.get("id") == session_id:
            s["status"] = "completed"
            break
    _write_index(idx)
    return jsonify({"ok": True})


def get_session_audio_dir(session_id: str) -> str:
    """
    Returns the audio directory path for a session.
    Creates the directory if it doesn't exist.
    """
    audio_dir = os.path.join(DATA_DIR, session_id, "audio")
    os.makedirs(audio_dir, exist_ok=True)
    return audio_dir


def set_last_audio_clip(session_id: str, audio_path: str) -> None:
    """
    Tracks the last generated audio clip for a session.
    Call this whenever a new audio clip is generated.
    """
    _session_last_audio[session_id] = audio_path
    
    # Also save to session JSON for persistence
    p = _session_path(session_id)
    if os.path.exists(p):
        with open(p, "r") as f:
            sess = json.load(f)
        sess["lastAudioClip"] = audio_path
        sess["lastAudioUpdated"] = datetime.utcnow().isoformat() + "Z"
        with open(p, "w") as f:
            json.dump(sess, f)


def get_last_audio_clip(session_id: str) -> str:
    """
    Retrieves the path to the last generated audio clip for a session.
    Returns None if no audio has been generated yet.
    """
    # Try in-memory first
    if session_id in _session_last_audio:
        audio_path = _session_last_audio[session_id]
        if os.path.exists(audio_path):
            return audio_path
    
    # Fall back to session JSON
    p = _session_path(session_id)
    if os.path.exists(p):
        with open(p, "r") as f:
            sess = json.load(f)
        audio_path = sess.get("lastAudioClip")
        if audio_path and os.path.exists(audio_path):
            return audio_path
    
    return None


@bp.post("/sessions/<session_id>/repeat")
def repeat_last_audio(session_id: str):
    """
    POST /api/sessions/<session_id>/repeat
    Returns the last generated audio clip for this session.
    Returns 404 if session doesn't exist or no audio has been generated.
    """
    # Validate session exists
    if not os.path.exists(_session_path(session_id)):
        return jsonify({"error": "session not found"}), 404
    
    # Get last audio clip
    audio_path = get_last_audio_clip(session_id)
    
    if audio_path is None:
        return jsonify({"error": "no audio generated yet"}), 404
    
    # Return the audio file
    return send_file(
        audio_path,
        mimetype="audio/wav",
        as_attachment=False,
        download_name=os.path.basename(audio_path)
    )

