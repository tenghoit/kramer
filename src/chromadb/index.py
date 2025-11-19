import chromadb
import logging
import logging
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction # type: ignore

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def get_collection(name: str) -> chromadb.Collection | None: # type: ignore
    client = chromadb.PersistentClient(path="/") # type: ignore
    collections = client.list_connections()
    if name in collections:
        logger.debug(f"Collection found: {name}")
        return client.get_collection(name=name)
    else:
        return None


def create_collection(name: str, model: str = "qwen3-embedding:8b") -> chromadb.Collection | None: # type: ignore
    client = chromadb.PersistentClient(path="/") # type: ignore
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
    return collection

