# Llama OCR RAG
A Deep Learning Project for Extracting Text from Images using PaddleOCR, Vector Store with FAISS, and Rag Chain Generation with LangGraph

## Overview
This project uses the PaddleOCR library to extract text from input images. The extracted text is then used to create a vector store with FAISS (Facebook's open-source similarity search library) and generate a rag chain with langgraph. Currently, this project only supports the Llama3.1 model.

### Table of Contents
* [Requirements](#requirements)
* [Usage](#usage)
* [Installation](#installation)
* [Dependencies](#dependencies)
* [Input/Output](#input-output)
* [Model Details](#model-details)

## Requirements

* Python 3.12
* PaddleOCR.
* FAISS.
* LangChain.
* Ollama embedding model: `nomic-embed-text`.

## Usage

1. Place your input images in the `images` directory.
2. Run the project using the following command: `python main.py`
3. The extracted chunks will be stored in the `chunks` directory in JSON format.

### Example Use Case
Suppose you have a folder `images` containing 5 image files:
```
images/
    img1.jpg
    img2.jpg
    img3.jpg
    img4.jpg
    img5.jpg
```

Running the project will generate corresponding JSON chunks in the chunks directory:

```
chunks/
    chunk1.json
    chunk2.json
    chunk3.json
    chunk4.json
    chunk5.json
```

## Installation

```
pip install -r requirements.txt
python main.py
```

## Input / Output

```
* Input Images in the `images/directory`
* Output: JSON chunks in the `chunks directory`
```

## Model Details

This project uses the Llama3.1 model, which is a type of large language model
pre-trained on a massive corpus of text data. The Ollama model nomic-embed-text
is used for text embeddings.
