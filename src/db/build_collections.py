from src.db.db_operations import add_lecture, create_collection, clear_collection, add_note
from src.text_extraction import extract_text
import sys
from pathlib import Path
import src.setup_logging



def main():
    # if len(sys.argv) != 2:
    #     print(f"Usage: python3 {sys.argv[0]}.py <file_path>")
    #     sys.exit(1)
    
    
    data_dir = Path(__file__).resolve().parents[2] / "data"
    class_code = "dsc360"
    topic = "conclusion"

    clear_collection("lectures")
    lecture_name = "Course Conclusion.pptx"
    lecture_path = data_dir / lecture_name
    for page, text in enumerate(extract_text(lecture_path)):
        add_lecture(class_code, topic, text, page)

    clear_collection("notes")
    note_name = "sample_note.txt"
    note_path = data_dir / note_name
    note_text = note_path.read_text()
    add_note(class_code, topic, note_text)


if __name__ == "__main__":
    main()
