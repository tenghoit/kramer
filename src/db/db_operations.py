import chromadb
import logging
import logging
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction # type: ignore
from src.text_extraction import extract_text
from pathlib import Path
import src.setup_logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

chromadb_dir = Path(__file__).resolve().parent / "chromadb"
client = chromadb.PersistentClient(path=chromadb_dir) 


def get_collection(name: str) -> chromadb.Collection | None:
    collections = client.list_collections()
    for c in collections:
        if c.name == name:  # check the name attribute
            logger.debug(f"Collection found: {name}")
            return client.get_collection(name=name)
    logger.debug(f"Collection not found: {name}")
    return None


def create_collection(name: str, model: str = "qwen3-embedding:8b") -> chromadb.Collection: 
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


def clear_collection(collection_name: str) -> chromadb.Collection:
    delete_collection(collection_name)
    return create_collection(collection_name)


def get_next_id(collection_name: str) -> str:
    collection = get_collection(collection_name) 
    ids = collection.get(limit=collection.count())["ids"]
    if not ids: return "1" # if empty
    last_id = ids[-1] # get() returns oldest to latest
    next_id = int(last_id) + 1
    return str(next_id)


def add_lecture(class_code: str, topic: str, text: str, page: int):
    lectures: chromadb.Collection = get_collection("lectures") # type: ignore
    metadata = {"class_code": class_code, "topic": topic, "page": page}
    lectures.add(
        ids=[get_next_id("lectures")],
        documents=[text],
        metadatas=[metadata]
    )
    logger.debug(f"{class_code} {topic} page {page} lecture added.")
    


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


def flatten_results(results: chromadb.QueryResult | chromadb.GetResult) -> list:  # type: ignore
    keys = results.keys()
    output = []
    for i in range(len(results[keys[0]])):
        item = dict()
        for key in keys:
            item[key] = results[key][i]
        output.append(item)
    return output




