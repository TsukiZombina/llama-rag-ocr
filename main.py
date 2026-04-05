# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## Packages

# %%
import os
import sys
import types

# Provide old import paths expected by paddlex:
# langchain.docstore.document -> Document
m1 = types.ModuleType("langchain.docstore.document")
from langchain_core.documents import Document

m1.Document = Document
sys.modules["langchain.docstore.document"] = m1

# langchain.text_splitter -> RecursiveCharacterTextSplitter
m2 = types.ModuleType("langchain.text_splitter")
from langchain_text_splitters import RecursiveCharacterTextSplitter

m2.RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter
sys.modules["langchain.text_splitter"] = m2

# %%
from langchain_ollama import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_classic.prompts import PromptTemplate
from paddleocr import PaddleOCR

from utils import create_vector_store, paddle_ocr_read_document, create_rag_chain, query_documents

# %%
IMAGE_DIR = "./images/"

# Initialize Spanish OCR model
ocr = PaddleOCR(lang='es')

for file in os.listdir(IMAGE_DIR):
    if os.path.isfile(IMAGE_DIR + file):
        paddle_ocr_read_document(ocr, IMAGE_DIR + file)

# %% [markdown]
# ## Create Vector Store

# %%
embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://localhost:11434"
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["", "", " ", ""]
)

vector_store = create_vector_store(text_splitter, embeddings, "./chunks")

# %% [markdown]
# ## Create RAG chain

# %%
llm = ChatOllama(model="llama3.1", temperature=1)

rag_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""Use the following context to answer the question.
Provide specific details and cite relevant information when possible.

Context: {context}

Question: {question}

Answer: Based on the provided context, here's what I found:"""
)

rag_chain = create_rag_chain(llm, rag_prompt, vector_store)

# %% [markdown]
# ## Execute Chain

# %%
queries = [
    "¿Qué doctores atendieron pacientes?",
    "¿Qué medicamentos fueron suministrados?",
    "¿Que me puedes decir de Farmacias del Ahorro?"
]

for query in queries:
    query_documents(rag_chain, query)
    print("-" * 50)

# %%
