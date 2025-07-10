import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

# Define desired file structure
list_of_files = [
    "app.py",
    "requirements.txt",
    "data/medical_docs.pdf",  # You can replace with an empty placeholder
    "rag/embedder.py",
    "rag/retriever.py",
    "rag/prompt.py",
    "llm/model.py",
    "templates/chat.html"
]

# Create each file/folder if not exists
for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir} for the file {filename}")

    if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
        with open(filepath, 'w') as f:
            if filename.endswith(".pdf"):
                pass  # don't write dummy content into a PDF
            else:
                f.write("")  # Create empty file
        logging.info(f"Creating empty file: {filepath}")
    else:
        logging.info(f"{filename} already exists")