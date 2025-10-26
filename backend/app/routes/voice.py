import os
import uuid
import json
from datetime import datetime
from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
from pydub import AudioSegment


bp = Blueprint("voice", __name__)


UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "uploads")
UPLOAD_DIR = os.path.abspath(UPLOAD_DIR)
os.makedirs(UPLOAD_DIR, exist_ok=True)


@bp.post("/voice/upload")
def upload_voice():
    audio = request.files.get("file")
    transcript = request.form.get("transcript", "").strip()
    duration_s = request.form.get("duration", "0").strip()

    if not audio:
        return jsonify({"error": "missing file"}), 400

    try:
        duration = float(duration_s)
    except Exception:
        duration = 0.0

    uid = str(uuid.uuid4())
    filename = f"{uid}.caf"
    audio_path = os.path.join(UPLOAD_DIR, filename)
    audio.save(audio_path)

    # convert to wav and delete
    # Load the .caf file
    audio = AudioSegment.from_file(audio_path, format="caf")
    # Export as .wav
    audio.export(os.path.join(UPLOAD_DIR, f"recording-{uid}.wav"), format="wav")


    words = len(transcript.split()) if transcript else 0
    wpm = round(words / (duration / 60.0), 2) if duration > 0 else None

    meta = {
        "id": uid,
        "filename": filename,
        "duration": duration,
        "transcript": transcript,
        "words": words,
        "wpm": wpm,
        "created_at": datetime.utcnow().isoformat() + "Z",
    }
    with open(os.path.join(UPLOAD_DIR, f"{uid}.json"), "w") as f:
        json.dump(meta, f)

    return jsonify({"id": uid, "wpm": wpm, "duration": duration, "words": words})


