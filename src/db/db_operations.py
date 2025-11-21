import chromadb
import logging
import logging
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction # type: ignore
from text_extraction import extract_text

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
client = chromadb.PersistentClient(path="/") # type: ignore


def get_collection(name: str) -> chromadb.Collection | None: # type: ignore
    collections = client.list_connections()
    if name in collections:
        logger.debug(f"Collection found: {name}")
        return client.get_collection(name=name)
    else:
        logger.debug(f"Collection not found: {name}")
        return None


def create_collection(name: str, model: str = "qwen3-embedding:8b") -> chromadb.Collection | None: # type: ignore
    """
    Create a new Chromadb collection.
    If collection already exists, return existing collection
    """
    collection = get_collection(name)
    if collection is not None: return collection

    logger.debug(f"Creating collection: {name}")
    collection = client.create_collection(
        name=name,
        embedding_function=OllamaEmbeddingFunction(
            url="http://localhost:11434",
            model_name=model,
        ),
        configuration={"hnsw": {"space": "cosine"}}
    )
    logger.debug(f"Collection created: {name}")
    return collection


def delete_collection(name: str):
    collection = get_collection(name)
    if collection is None:
        logger.debug(f"Deletion Failed. Collection does not exist: {name}")
        return
    client.delete_collection(name=name)
    logger.debug(f"Collection deleted: {name}")
    return


def clear_collection(collection_name: str):
    delete_collection(collection_name)
    create_collection(collection_name)


def get_next_id(collection_name: str) -> int:
    collection = get_collection(collection_name)
    ids = collection.get(limit=collection.count())["ids"]
    if not ids: return 1 # if empty
    last_id = ids[-1] # get() returns oldest to latest
    next_id = int(last_id) + 1
    return next_id


def add_note(class_code: str, topic: str, file_path):
    notes: chromadb.Collection = get_collection("notes") # type: ignore
    document: str = extract_text(file_path)
    metadata = {"class_code": class_code, "topic": topic}
    notes.add(
        ids=[get_next_id("notes")],
        documents=[document],
        metadatas=[metadata]
    )
    logger.debug(f"{class_code} {topic} notes added.")
    return


def get_note(class_code: str, topic: str):
    notes: chromadb.Collection = get_collection("notes") # type: ignore
    metadata = {
        "$and": [
            {"class_code": class_code},
            {"topic": topic}
        ]
    }
    results = notes.query(
        query_embeddings=[],
        where=metadata
    )
    

