import json
import ollama
import logging
import setup_logging
from pathlib import Path
import numpy as np
import json
from slide_chunker import chunkBySlide as ChunkBySlide
from pydantic import BaseModel
from text_extraction import extract_text, pptx_to_texts


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


notes = get_notes()


def add_note(class_code: str, topic: str, text: str):
    keywords = get_keywords(text)
    embedding_text = "; ".join(keywords)

    item = {
        "class_code": class_code,
        "topic": topic,
        "text": text,
        "keywords": keywords,
        "embedding": get_embedding(embedding_text)
    }

    notes.append(item)
    with open(notes_path, "w") as f:
        json.dump(notes, f, indent=4)
    logger.debug(f"{class_code} {topic} notes added.")


def query_notes(class_code: str, topic: str) -> dict | None:
    for note in notes:
        if note["class_code"] == class_code and note["topic"] == topic:
            return note
    return None


def clear_notes():
    notes_path.write_text("[]")
    global notes
    notes = get_notes()
    logger.debug(f"notes cleared")


def get_lectures() -> list[dict]:
    with open(lectures_path, "r") as f:
        lectures = json.load(f)
    return lectures


lectures = get_lectures()


def add_lecture(class_code: str, topic: str, page: int, text: str):
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
    logger.debug(f"{class_code} {topic} page {page} lecture added.")


def query_lectures(class_code: str, topic: str) -> list[dict]:
    output = []
    for lecture in lectures:
        if lecture["class_code"] == class_code and lecture["topic"] == topic:
            output.append(lecture)
    return output


def clear_lectures():
    lectures_path.write_text("[]")
    global lectures
    lectures = get_lectures()
    logger.debug(f"lectures cleared")


def clear_db():
    clear_notes()
    clear_lectures()


def is_content(topic: str, text: str, model: str = "gemma3:12b") -> bool:
    if len(text) == 0: return False

    class Answer(BaseModel):
        is_content: bool

    system_message = """
    You are a content classifier for slide text.
    Your job is to determine whether the given text contains meaningful lecture content,
    or if it is non-content (titles, headers, slide numbers, footers, decorative text,
    irrelevant short fragments, or noise).

    Guidelines:
    - "Content" means real explanatory material, definitions, bullets, descriptions, examples,
      or anything conceptually meaningful.
    - NOT content: very short text (<3 words) unless clearly a concept name;
      e.g., 'Agenda', 'Conclusion', 'CSC 360', 'Unit 9', slide numbers, dates,
      class codes, instructor names, section titles, etc.

    Output must strictly follow the JSON schema.
    """

    user_prompt = f"""
    Determine if this slide text contains meaningful lecture content.
    Topic of the lecture: {topic}
    Slide text:
    {text}
    """

    result = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt}
        ],
        format=Answer.model_json_schema(),
        options={"temperature": 0}
    )

    content = result["message"]["content"]
    result = Answer.model_validate_json(content).is_content
    logger.debug(f"{result} | {text}")
    return result
    

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
    content = result["message"]["content"]
    keywords = Keywords.model_validate_json(content).keywords
    logger.debug(f"Keywords: {keywords}")
    return keywords


def get_cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    v1 = np.array(vec1, dtype=float)
    v2 = np.array(vec2, dtype=float)

    dot = np.dot(v1, v2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)

    if norm == 0:
        return 0.0

    return float(dot / norm)


def cmp(class_code: str, topic: str) -> list[dict]:
    note = query_notes(class_code, topic)
    lectures = query_lectures(class_code, topic)

    if not note: raise ValueError(F"Missing note for {class_code} {topic}")
    if not lectures: raise ValueError(F"Missing lectures for {class_code} {topic}")

    threshold = 0.7
    missing_lectures = []

    for lecture in lectures:
        cosine_similarity = get_cosine_similarity(note["embedding"], lecture["embedding"])
        logger.debug(f"{cosine_similarity}")
        if cosine_similarity < threshold:
            missing_lectures.append(lecture)

    return missing_lectures


def generate_recommendation(note: dict, missing_lectures: list[dict], model: str = "gemma3:12b") -> str:
    note_text = note.get("text", "").strip()

    if not missing_lectures:
        return "Your notes cover the lecture content well. No missing areas detected."

    missing_block = "\n\n---\n\n".join(
        f"Slide {lec.get('page', '?')}:\n{lec.get('text', '').strip()}"
        for lec in missing_lectures
        if lec.get("text", "")
    )

    system_message = """
    You are a teaching assistant reviewing a student's notes.
    Identify what important concepts the student failed to include,
    based on the provided lecture material.
    Explain what they should add or review in a clear, concise way.
    Do not rewrite full slides. Summarize the gaps.
    """

    user_prompt = f"""
    STUDENT NOTES:
    {note_text}

    MISSING LECTURE CONTENT:
    {missing_block}

    Provide a helpful recommendation about what the student should add or review.
    """

    resp = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt}
        ],
        options={"temperature": 0.2}
    )
    print(resp["message"]["content"].strip())
    return resp["message"]["content"].strip()
    


def embed_all_lectures():
    lectures_dir = Path(__file__).resolve().parents[1] / "data/lectures/"
    class_code = "dsc360"
    topics = {
        "LLM-Assisted Data Extraction.pptx": "data extraction",
        "Slides Safe Execution.pptx": "safe execution",
        "Course Conclusion.pptx": "conclusion",
        "Transformer Architecture.pptx": "transformer architecture",
        "Slides RAG Revisited.pptx": "rag",
        "Slides What is an Agent.pptx": "agent",
        "Agents in Practice.pptx": "practical agent",
        "Ship It Safely - Choosing and Justifying Your LLM Deployment.pptx": "deployment",
    }

    file_paths = list(topics.keys())
    for file_path in file_paths:
        full_path = lectures_dir / file_path
        topic = topics[file_path]

        for raw_text in pptx_to_texts(full_path):
            slide_chunks = ChunkBySlide(raw_text)
        
            for chunk in slide_chunks:
                text = chunk["content"]
                page = chunk["slide_number"]
            
                if is_content(topic=topic, text=text):
                    add_lecture(class_code, topic, page, text)


        logger.debug(f"embedded {file_path}")




def main():
    clear_db()
    embed_all_lectures()


if __name__ == "__main__":
    main()



    