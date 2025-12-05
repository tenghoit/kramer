import json
import ollama
import logging
import setup_logging
import argparse
import sys
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
    # print(resp["message"]["content"].strip())
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



def embed_notes():
    clear_notes()
    notes_dir = Path(__file__).resolve().parents[1] / "data/notes/"
    class_code = "dsc360"

    topics = {
        "safe_execution_notes.txt": "safe execution",
        "Trandsformer_architecture.txt": "transformer architecture",
        "conclusion.txt": "conclusion",
        "extraction_notes.txt": "data extraction",
        "agent_in_practice.txt": "practical agent",
        "ship_safely_notes.txt": "deployment",
        "rag_revisited_notes.txt": "rag",
        "what_is_agent_notes.txt": "agent",
    }
    file_paths = list(topics.keys())
    for file_path in file_paths:
        full_path = notes_dir / file_path
        topic = topics[file_path]
        text = extract_text(full_path)
        add_note(class_code=class_code, topic=topic, text=text)
        logger.debug(f"Added [{file_path}] to notes")


def main():
    embed_notes()

def cli():
    parser = argparse.ArgumentParser(prog="notes-agent", description="Compare lecture slides to student notes and generate recommendations.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_init = subparsers.add_parser("init-db", help="Clear DB and embed all lectures.")
    p_init.set_defaults(func=cmd_init_db)

    p_add = subparsers.add_parser("add-note", help="Add a note for a class and topic.")
    p_add.add_argument("class_code", type=str)
    p_add.add_argument("topic", type=str)
    p_add.add_argument("--text", type=str, help="Note text. If omitted, read from stdin.")
    p_add.add_argument("--file", type=str, help="Path to file containing note text.")
    p_add.set_defaults(func=cmd_add_note)

    p_cmp = subparsers.add_parser("compare", help="Show missing lecture chunks for a class and topic.")
    p_cmp.add_argument("class_code", type=str)
    p_cmp.add_argument("topic", type=str)
    p_cmp.set_defaults(func=cmd_compare)

    p_rec = subparsers.add_parser("recommend", help="Generate a recommendation based on missing content.")
    p_rec.add_argument("class_code", type=str)
    p_rec.add_argument("topic", type=str)
    p_rec.set_defaults(func=cmd_recommend)

    p_slide = subparsers.add_parser("add-slide", help="Add a single lecture slide chunk manually.")
    p_slide.add_argument("class_code", type=str)
    p_slide.add_argument("topic", type=str)
    p_slide.add_argument("page", type=int)
    p_slide.add_argument("--text", type=str, help="Slide text. If omitted, read from stdin.")
    p_slide.add_argument("--file", type=str, help="Path to a file containing slide text.")
    p_slide.set_defaults(func=cmd_add_slide)


    p_clear = subparsers.add_parser("clear-db", help="Clear notes and lectures.")
    p_clear.set_defaults(func=cmd_clear_db)

    args = parser.parse_args()
    args.func(args)


def cmd_init_db(args):
    clear_db()
    embed_all_lectures()
    print("Database cleared and lectures embedded.")


def cmd_add_note(args):
    class_code = args.class_code
    topic = args.topic

    if args.text is not None:
        text = args.text
    elif args.file is not None:
        text = Path(args.file).read_text(encoding="utf-8")
    else:
        text = sys.stdin.read()

    add_note(class_code, topic, text)
    print(f"Added note for {class_code} {topic}.")


def cmd_compare(args):
    class_code = args.class_code
    topic = args.topic
    try:
        missing = cmp(class_code, topic)
    except ValueError as e:
        print(str(e))
        sys.exit(1)

    if not missing:
        print("No missing lecture chunks detected for this topic.")
        return

    print(f"Missing lecture chunks for {class_code} {topic}:")
    for lec in missing:
        page = lec.get("page", "?")
        text = lec.get("text", "").strip().replace("\n", " ")
        print(f"\n--- Slide {page} ---")
        print(text)


def cmd_recommend(args):
    class_code = args.class_code
    topic = args.topic

    note = query_notes(class_code, topic)
    if not note:
        print(f"No note found for {class_code} {topic}.")
        sys.exit(1)

    try:
        missing = cmp(class_code, topic)
    except ValueError as e:
        print(str(e))
        sys.exit(1)

    rec = generate_recommendation(note, missing)
    print(rec)

def cmd_add_slide(args):
    class_code = args.class_code
    topic = args.topic
    page = args.page

    if args.text is not None:
        text = args.text
    elif args.file is not None:
        text = Path(args.file).read_text(encoding="utf-8")
    else:
        text = sys.stdin.read()

    add_lecture(class_code, topic, page, text)
    print(f"Added lecture slide for {class_code} {topic} page {page}.")
    
def cmd_clear_db(args):
    clear_db()
    print("Database cleared.")
    
if __name__ == "__main__":
    cli()




    
