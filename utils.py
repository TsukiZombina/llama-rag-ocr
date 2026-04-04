import json
from typing import Any, Dict, List

from paddleocr import PaddleOCR
from pathlib import Path


def paddle_ocr_read_document(ocr: PaddleOCR, image_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Reads and scans an image from the given path and uses PaddleOCR to extract
    the text. The extracted information is stored in the `chunks` directory.

    Returns a dictionary of dictionaries, where each element contains:
    - "text" the extracted text string.
    - "confidence" The recognition confidence score.
    """
    result = ocr.predict(image_path)
    page = result[0]

    texts = page['rec_texts']
    boxes = page['dt_polys']
    scores = page.get('rec_scores', [None] * len(texts))

    extracted_items = {}
    for index, (text, box, score) in enumerate(zip(texts, boxes, scores)):
        item = {
            'text': text,
        }
        if score is not None:
            item['confidence'] = score

        extracted_items[f"item_{index}"] = (item)

    with open("./chunks/" + Path(image_path).stem + ".json", "w") as json_file:
        json.dump(extracted_items, json_file)

    return extracted_items
