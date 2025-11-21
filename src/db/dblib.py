import chromadb
import logging
import logging
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction # type: ignore

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


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


def get_notes(class_code: str, topic: str):
    notes = get_collection("notes")
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
    

