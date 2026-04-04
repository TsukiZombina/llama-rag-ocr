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
from utils import paddle_ocr_read_document
from paddleocr import PaddleOCR

# %%
IMAGE_DIR = "./images/"

# Initialize Spanish OCR model
ocr = PaddleOCR(lang='es')

for file in os.listdir(IMAGE_DIR):
    if os.path.isfile(IMAGE_DIR + file):
        paddle_ocr_read_document(ocr, IMAGE_DIR + file)

# %%
