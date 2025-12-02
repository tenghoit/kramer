from src.db.db_operations import add_lecture, create_collection, clear_collection
from src.text_extraction import extract_text
import sys
from pathlib import Path
import src.setup_logging

class_code = "dsc360"

def main():
    # if len(sys.argv) != 2:
    #     print(f"Usage: python3 {sys.argv[0]}.py <file_path>")
    #     sys.exit(1)
    
    data_dir = Path(__file__).resolve().parents[2] / "data"
    file_name = "LLM-Assisted Data Extraction.pptx"
    file_path = data_dir / file_name
    topic = "data extraction"

    clear_collection("lectures")

    for page, text in enumerate(extract_text(file_path)):
        add_lecture(class_code, topic, text, page)


if __name__ == "__main__":
    main()
