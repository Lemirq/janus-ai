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

# quick little cli: delete all docs, peek docs
if __name__ == "__main__":
    print("ChromaDB CLI. Commands:")
    print(" 1 - peek   - show a sample of docs in the collection")
    print(" 2 - delete - delete all docs in the collection")
    print(" 3 - quit   - exit CLI")
    while True:
        command = input("Enter command: ")
        if command == "1":
            print(collection.peek())
        elif command == "2":
            ids = collection.peek().get("ids")
            if ids:
                collection.delete(ids=ids)
            else:
                print("No documents to delete.")
            print("Deleted all docs")
        elif command == "3":
            print("Exiting.")
            break
        else:
            print("Invalid command. Use '1', '2', or '3'.")