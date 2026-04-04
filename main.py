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

# %%
from utils import paddle_ocr_read_document

# %%
IMAGE_DIR = "./images"

for file in os.listdir(IMAGE_DIR):
    if os.path.isfile(IMAGE_DIR + file):
        paddle_ocr_read_document(IMAGE_DIR + file)

# %%
