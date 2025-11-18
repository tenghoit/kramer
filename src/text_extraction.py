from pypdf import PdfReader, errors
from pathlib import Path


def pdf_to_text(file_path: Path) -> str:
    try:
        reader = PdfReader(file_path)
        text = "\n".join(page.extract_text() or "" for page in reader.pages) # or "" b/c reader return None if image
        text = re.sub(r"[\n]{2,}", "\n", text) # Replace multiple newlines with a single newline
        text = re.sub(r"[\s]+", " ", text) # Replace multiple spaces with a single space
        return text 
    except errors.PyPdfError as e:
        raise ValueError(f"Failed to read PDF: {file_path}") from e