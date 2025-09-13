import os
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
import glob
from PyPDF2 import PdfReader

# Load API keys
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_INDEX = "ai-study-buddy"

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create index if not exists
if PINECONE_INDEX not in [i["name"] for i in pc.list_indexes()]:
    pc.create_index(
        name=PINECONE_INDEX,
        dimension=1536,   # for text-embedding-3-small
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(PINECONE_INDEX)

# Initialize Groq client
client = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")

# --- Function to load documents from /data folder ---
def load_documents():
    docs = []
    
    # Load TXT and MD files
    for file in glob.glob("data/*.txt") + glob.glob("data/*.md"):
        with open(file, "r", encoding="utf-8") as f:
            docs.append(f.read())
    
    # Load PDFs
    for file in glob.glob("data/*.pdf"):
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        docs.append(text)
    
    return docs

# --- Process documents ---
documents = load_documents()
vectors = []
for i, doc in enumerate(documents):
    emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=doc
    )
    vector = emb.data[0].embedding

    vectors.append({
        "id": f"doc-{i}",
        "values": vector,
        "metadata": {"text": doc[:200]}  # store preview only
    })

# Upsert into Pinecone
index.upsert(vectors)

print(f"✅ {len(documents)} documents added to Pinecone index '{PINECONE_INDEX}'")
