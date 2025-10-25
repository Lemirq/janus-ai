import os


CHROMA_PERSIST_PATH = os.getenv(
    "CHROMA_PERSIST_PATH",
    "/Users/vs/Coding/janus-ai/backend/chroma_data",
)
DEFAULT_COLLECTION_NAME = os.getenv("CHROMA_DEFAULT_COLLECTION", "docs")


