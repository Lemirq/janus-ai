import os
import json
import uuid
from datetime import datetime
from flask import Blueprint, request, jsonify


bp = Blueprint("sessions", __name__)


DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "data", "sessions"))
os.makedirs(DATA_DIR, exist_ok=True)
INDEX_PATH = os.path.join(DATA_DIR, "index.json")


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


