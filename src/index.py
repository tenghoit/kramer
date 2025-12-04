import json
import ollama
import logging
import setup_logging
from pathlib import Path
import numpy as np
import json
from pydantic import BaseModel


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

db_dir = Path('db')
notes_path = db_dir / "notes.json"
lectures_path = db_dir / "lectures.json"

def normalize(vector):
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm
    return vector.tolist()


def get_embedding(text: str, model: str = "qwen3-embedding:8b"):
    result = ollama.embed(model, text)
    embedding = result["embeddings"][0]
    vector = np.array(embedding, dtype=float)
    return normalize(vector)


def get_notes() -> list[dict]:
    with open(notes_path, "r") as f:
        notes = json.load(f)
    return notes


def add_note(class_code: str, topic: str, text: str):
    notes = get_notes()
    item = {
        "class_code": class_code,
        "topic": topic,
        "text": text,
        "embedding": get_embedding(text)
    }
    notes.append(item)
    with open(notes_path, "w") as f:
        json.dump(notes, f, indent=4)


def get_note(class_code: str, topic: str) -> dict | None:
    notes = get_notes()
    for note in notes:
        if note["class_code"] == class_code and note["topic"] == topic:
            return note
    return None


def get_keywords(text: str, model: str = "gemma3:12b") -> list[str]:

    class Keywords(BaseModel):
        keywords: list[str]

    system_message = """
    You are a keyword extraction engine.
    Your job is to extract the core technical concepts from the input text.
    Return only concise, meaningful keywords suitable for semantic comparison.
    Avoid long phrases, sentences, or non-essential filler.
    Do NOT explain anything.
    Output must strictly match the JSON schema.
    """

    prompt = f"""
    Extract the main keywords from the following text.
    Focus on the important concepts only:

    TEXT:
    {text}
    """


    result = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ],
        format=Keywords.model_json_schema(),
        options={"temperature": 0}
    )
    content = result.message.content or ''
    return Keywords.model_validate_json(content).keywords


def get_lectures() -> list[dict]:
    with open(lectures_path, "r") as f:
        lectures = json.load(f)
    return lectures


def add_lecture(class_code: str, topic: str, page: int, text: str):
    lectures = get_lectures()
    keywords = get_keywords(text)
    embedding_text = "; ".join(keywords)

    item = {
        "class_code": class_code,
        "topic": topic,
        "page": page, 
        "text": text,
        "keywords": keywords,
        "embedding": get_embedding(embedding_text)
    }

    lectures.append(item)
    with open(lectures_path, "w") as f:
        json.dump(lectures, f, indent=4)
    




    