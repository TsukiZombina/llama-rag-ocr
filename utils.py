import json
import os
from pathlib import Path
from typing import Any, Dict, List

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA
from paddleocr import PaddleOCR


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


def create_vector_store(text_splitter, embeddings: OllamaEmbeddings, input_path: str):
    documents = []

    # Load documents from directory
    for filename in os.listdir(input_path):
        if filename.endswith(".json"):
            with open(os.path.join(input_path, filename), "r") as file:
                content = file.read()
                documents.append(content)

    # Split documents into chunks
    texts = text_splitter.create_documents(documents)

    # Create store
    vector_store = FAISS.from_documents(texts, embeddings)

    return vector_store


def create_rag_chain(llm, prompt, vector_store):
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}  # Retrieve top 4 relevant chunks
    )

    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )

    return rag_chain


def query_documents(rag_chain, question):
    result = rag_chain({"query": question})

    print(f"Question: {question}")
    print(f"Answer: {result['result']}")
    print(f"Sources: {len(result['source_documents'])} documents referenced")

    return result
