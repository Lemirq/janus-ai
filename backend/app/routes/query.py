from flask import Blueprint, request, jsonify
from ..vectorstore import get_collection


bp = Blueprint("query", __name__)


@bp.get("/query")
def query():
    q = request.args.get("q") or (request.get_json(silent=True) or {}).get("q")
    k = int(request.args.get("top_k", 5))
    collection_name = request.args.get("collection")
    col = get_collection(collection_name) if collection_name else get_collection()

    if not q:
        return jsonify({"error": "missing q"}), 400

    res = col.query(query_texts=[q], n_results=k)
    results = []
    for i in range(len(res.get("ids", [[]])[0])):
        results.append({
            "id": res["ids"][0][i],
            "text": res["documents"][0][i],
            "metadata": (res.get("metadatas") or [[{}]])[0][i],
            "distance": (res.get("distances") or [[None]])[0][i],
        })
    return jsonify({"results": results})


