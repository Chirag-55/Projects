# rag/embedder.py
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
from pinecone import Pinecone, ServerlessSpec
import os
import glob
import pytesseract
from PIL import Image

class MedicalEmbedder:
    def __init__(self, pdf_path):
        # Load sentence transformer model
        self.model = SentenceTransformer("all-MiniLM-L6-v2")  # dimension = 768

        # Initialize Pinecone client
        self.pc = Pinecone(api_key="YOUR_API_KEY")
        index_name = "medical-bot-index"

        # Create or connect to Pinecone index
        if index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=index_name,
                dimension=768,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )

        self.index = self.pc.Index(index_name)

        # Load PDF and store chunks
        self.text_chunks = self._load_all_pdfs_chunks("data/")

        # Add chunks to Pinecone if empty
        if self.index.describe_index_stats()['total_vector_count'] == 0:
            print("Index is empty. Adding PDF chunks...")
            embeddings = self.model.encode(self.text_chunks).tolist()
            vectors = [(f"id-{i}", vec, {"text": chunk}) for i, (vec, chunk) in enumerate(zip(embeddings, self.text_chunks))]
            self._batch_upsert(vectors)
        else:
            print("Using existing Pinecone index.")

    def _batch_upsert(self, vectors, batch_size=100):
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            self.index.upsert(vectors=batch)

    def _load_pdf_chunks(self, pdf_path, chunk_size=200):
        reader = PdfReader(pdf_path)
        full_text = " ".join([page.extract_text() or "" for page in reader.pages])
        words = full_text.split()
        chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
        return chunks

    def _load_all_pdfs_chunks(self, folder_path, chunk_size=200):
        all_chunks = []
        for pdf_path in glob.glob(os.path.join(folder_path, "*.pdf")):
            reader = PdfReader(pdf_path)
            full_text = " ".join([page.extract_text() or "" for page in reader.pages])
            words = full_text.split()
            chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
            all_chunks.extend(chunks)
        return all_chunks

    def search(self, query, top_k=3):
        query_vec = self.model.encode([query])[0].tolist()
        result = self.index.query(vector=query_vec, top_k=top_k, include_metadata=True)
        return [match['metadata']['text'] for match in result['matches']]

    # Auto-Triage Agent: detect symptoms and ask follow-ups
    def detect_symptoms_and_followup(self, user_input):
        common_symptoms = ["fever", "cough", "vomiting", "fatigue", "chest pain", "headache"]
        for symptom in common_symptoms:
            if symptom in user_input.lower():
                return f"You mentioned '{symptom}'. Can you describe how long you've had this symptom and its severity?"
        return None

    # Multimodal Medical Report Analyzer: OCR from image and summarize
    def analyze_medical_image(self, image_path):
        try:
            img = Image.open(image_path)
            text = pytesseract.image_to_string(img)
            if text.strip():
                summary_prompt = f"Summarize this medical report text in simple terms:\n{text}"
                from llm.model import ask_llm
                return ask_llm(summary_prompt)
            else:
                return "Could not extract any readable text from the image."
        except Exception as e:
            return f"Error processing image: {str(e)}"
