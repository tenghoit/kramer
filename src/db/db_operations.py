import chromadb
import logging
import logging
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction 
from src.text_extraction import extract_text
from pathlib import Path
import src.setup_logging
from numpy import dot
from numpy.linalg import norm


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

chromadb_dir = Path(__file__).resolve().parent / "chromadb"
client = chromadb.PersistentClient(path=chromadb_dir) 


class Result:
    def __init__(self, id: str, embedding: list, document: str, metadata: dict, distance: float | None = None) -> None:
        self.id = id
        self.embedding = embedding
        self.document = document
        self.metadata = metadata
        self.distance = distance


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
        embedding_function=OllamaEmbeddingFunction(model_name=model), # type: ignore
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
    if collection is None: raise ValueError(f"Nonexisting collection") 
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


def get_lectures(class_code: str, topic: str) -> list[Result]:
    lectures: chromadb.Collection = get_collection("lectures") # type: ignore
    metadata = {
        "$and": [
            {"class_code": class_code},
            {"topic": topic}
        ]
    }
    result = lectures.get(
        where=metadata,
        limit=lectures.count(),
        include=["metadatas", "documents", "embeddings"]
    )
    flat_result = flatten_get_result(result)
    return flat_result


def add_note(class_code: str, topic: str, text: str):
    notes: chromadb.Collection = get_collection("notes") # type: ignore
    document: str = text
    metadata = {"class_code": class_code, "topic": topic}
    notes.add(
        ids=[get_next_id("notes")],
        documents=[document],
        metadatas=[metadata]
    )
    logger.debug(f"{class_code} {topic} notes added.")
    return


def get_note(class_code: str, topic: str) -> Result:
    notes: chromadb.Collection = get_collection("notes") # type: ignore
    metadata = {
        "$and": [
            {"class_code": class_code},
            {"topic": topic}
        ]
    }
    result = notes.get(
        where=metadata,
        limit=notes.count(),
        include=["metadatas", "documents", "embeddings"]
    )
    flat_result = flatten_get_result(result)
    return flat_result[0]


def flatten_get_result(result: chromadb.GetResult) -> list[Result]:  # type: ignore
    output = []
    for i in range(len(result["ids"])):
        item = Result(
            id=result["ids"][i],
            embedding=result["embeddings"][i],
            document=result["documents"][i],
            metadata=result["metadatas"][i],
        )
        output.append(item)
    return output


def cmp_note_lecture(class_code: str, topic: str):
    relevant_lectures = get_lectures(class_code, topic)
    lecture_embeddings = [item.embedding for item in relevant_lectures]

    relevant_note = get_note(class_code, topic)
    notes: chromadb.Collection = get_collection("notes") # type: ignore
    
    result = notes.query(
        query_embeddings=lecture_embeddings,
        ids=[relevant_note.id]
    )

    threshold = 0.5
    missing_lectures = []

    def cosine_distance(a, b):
        return 1 - dot(a, b) / (norm(a) * norm(b))

    for i, lecture in enumerate(relevant_lectures):
        distance = cosine_distance(lecture.embedding, relevant_note.embedding)
        if distance < threshold:
            missing_lectures.append(lecture)

    return missing_lectures
    

def main():
    collections = client.list_collections()
    for c in collections: print(c.name)
    for item in get_lectures("dsc360", "data extraction"):
        print(f"{item.id} | {item.metadata} | {item.document}")



if __name__ == "__main__":
    main()


