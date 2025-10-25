import uuid
from flask import Blueprint, request, jsonify
from ..vectorstore import get_collection


bp = Blueprint("ingest", __name__)


@bp.post("/ingest")
def ingest():
    payload = request.get_json(force=True, silent=True) or {}
    items = payload.get("documents", [])
    collection_name = payload.get("collection")
    col = get_collection(collection_name) if collection_name else get_collection()

    texts, ids, metadatas = [], [], []
    for item in items:
        if isinstance(item, str):
            texts.append(item)
            ids.append(str(uuid.uuid4()))
            metadatas.append(None)
        elif isinstance(item, dict):
            texts.append(item.get("text", ""))
            ids.append(item.get("id") or str(uuid.uuid4()))
            meta = item.get("metadata")
            metadatas.append(meta if meta else None)
    if not texts:
        return jsonify({"added": 0}), 200

    # If all metadata entries are None, omit metadatas entirely to satisfy Chroma validation
    if any(m is not None for m in metadatas):
        col.add(documents=texts, ids=ids, metadatas=metadatas)
    else:
        col.add(documents=texts, ids=ids)
    return jsonify({"added": len(texts)})


