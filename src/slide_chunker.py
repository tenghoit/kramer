import re
from typing import List, Dict

def chunkBySlide(text: str) -> List[Dict[str, str]]:
    lines = text.splitlines()
    slideBoundary = re.compile(r"^\s*Slide\s+(\d+)", re.IGNORECASE)

    slides = []
    currentSlideNumber = None
    currentSlideLines = []

    for line in lines:
        match = slideBoundary.match(line)
        if match:
            if currentSlideNumber is not None:
                title = next((l.strip() for l in currentSlideLines if l.strip()), "")
                slides.append({
                    "slide_number": currentSlideNumber,
                    "title": title,
                    "content": "\n".join(currentSlideLines).strip()
                })

            currentSlideNumber = int(match.group(1))
            currentSlideLines = []
        else:
            currentSlideLines.append(line)

    if currentSlideNumber is not None:
        title = next((l.strip() for l in currentSlideLines if l.strip()), "")
        slides.append({
            "slide_number": currentSlideNumber,
            "title": title,
            "content": "\n".join(currentSlideLines).strip()
        })

    return slides

