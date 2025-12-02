from src.db.db_operations import add_lecture, create_collection
from src.text_extraction import extract_text
import sys
from pathlib import Path
import src.setup_logging

class_code = "dsc360"

def main():
    # if len(sys.argv) != 2:
    #     print(f"Usage: python3 {sys.argv[0]}.py <file_path>")
    #     sys.exit(1)


    pure_file_path = Path("/Users/tenghoitkouch/Programming/kramer/data/LLM-Assisted Data Extraction.pptx")
    topic = "data extraction"

    create_collection("lectures")

    for page, text in enumerate(extract_text(pure_file_path)):
        add_lecture(class_code, topic, text, page)


if __name__ == "__main__":
    main()
