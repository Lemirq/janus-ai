import chromadb
from chromadb.utils.embedding_functions import (
    SentenceTransformerEmbeddingFunction,
)

from app.config import CHROMA_PERSIST_PATH, DEFAULT_COLLECTION_NAME


_client = chromadb.PersistentClient(path=CHROMA_PERSIST_PATH)


# peek collection
collection = _client.get_collection(DEFAULT_COLLECTION_NAME)

# ids = collection.peek().get("ids")
# collection.delete(ids=ids)

print(collection.peek())