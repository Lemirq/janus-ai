import chromadb
from chromadb.utils.embedding_functions import (
    SentenceTransformerEmbeddingFunction,
)

from .config import CHROMA_PERSIST_PATH, DEFAULT_COLLECTION_NAME


_embedding_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
_client = chromadb.PersistentClient(path=CHROMA_PERSIST_PATH)


def get_collection(name: str = DEFAULT_COLLECTION_NAME):
    return _client.get_or_create_collection(name=name, embedding_function=_embedding_fn)


