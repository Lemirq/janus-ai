import uuid
from flask import Blueprint, request, jsonify
from ..vectorstore import get_collection
from .sessions import _session_path
import os
import json


bp = Blueprint("ingest", __name__)


@bp.post("/ingest")
def ingest():
    payload = request.get_json(force=True, silent=True) or {}
    items = payload.get("documents", [])
    collection_name = payload.get("collection")
    session_id = payload.get("sessionId")
    col = get_collection(collection_name) if collection_name else get_collection()

    texts, ids, metadatas = [], [], []
    for item in items:
        if isinstance(item, str):
            texts.append(item)
            ids.append(str(uuid.uuid4()))
            meta = {"source": "upload"}
            if session_id:
                meta["sessionId"] = session_id
            metadatas.append(meta)
        elif isinstance(item, dict):
            texts.append(item.get("text", ""))
            ids.append(item.get("id") or str(uuid.uuid4()))
            meta = item.get("metadata") or {}
            if session_id:
                meta.setdefault("sessionId", session_id)
            metadatas.append(meta if meta else None)
    if not texts:
        return jsonify({"added": 0}), 200

    # If all metadata entries are None, omit metadatas entirely to satisfy Chroma validation
    if any(m is not None for m in metadatas):
        col.add(documents=texts, ids=ids, metadatas=metadatas)
    else:
        col.add(documents=texts, ids=ids)

    # Persist file IDs to the session JSON for later retrieval
    if session_id:
        try:
            sess_path = _session_path(session_id)
            if os.path.exists(sess_path):
                with open(sess_path, 'r') as f:
                    sess = json.load(f)
                existing = set(sess.get("fileIds") or [])
                for _id in ids:
                    existing.add(_id)
                sess["fileIds"] = list(existing)
                with open(sess_path, 'w') as f:
                    json.dump(sess, f)
        except Exception:
            # Non-fatal if session file missing or invalid
            pass

    return jsonify({"added": len(texts), "ids": ids})


