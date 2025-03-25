import os
import json
import torch
import faiss
import numpy as np
import PyPDF2
import docx
import pandas as pd
import pytesseract
import cv2
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Initialize Flask app
app = Flask(__name__)
CORS(app)

try:
    t5_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")  
    t5_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
except Exception as e:
    print(f"Model loading error: {e}")
    t5_model = None
    t5_tokenizer = None

try:
    retrieval_model = SentenceTransformer("all-MiniLM-L6-v2") 
except Exception as e:
    print(f"Retrieval model loading error: {e}")
    retrieval_model = None

# FAISS index setup
d = 384  # Embedding dimension
faiss_index = faiss.IndexFlatL2(d)
documents = []  # Store document texts
uploaded_file_path = None  # Track uploaded file

# -------------------- TEXT EXTRACTION --------------------

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF files."""
    try:
        text = ""
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() or ""
        return text.strip()
    except Exception as e:
        print(f"PDF extraction error: {e}")
        return ""

def extract_text_from_docx(docx_path):
    """Extract text from DOCX files."""
    try:
        doc = docx.Document(docx_path)
        return "\n".join([para.text.strip() for para in doc.paragraphs if para.text.strip()])
    except Exception as e:
        print(f"DOCX extraction error: {e}")
        return ""

def extract_text_from_csv(csv_path):
    """Extract text from CSV files."""
    try:
        df = pd.read_csv(csv_path)
        return df.to_string(index=False)
    except Exception as e:
        print(f"CSV extraction error: {e}")
        return ""

def extract_text_from_image(image_path):
    """Extract text from images using OCR."""
    try:
        image = Image.open(image_path)
        return pytesseract.image_to_string(image)
    except Exception as e:
        print(f"Image extraction error: {e}")
        return ""

def extract_text_from_video(video_path):
    """Extract text from video frames using OCR."""
    try:
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        text = ""

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame_count > 10: 
                break
            text += pytesseract.image_to_string(frame)
            frame_count += 1

        cap.release()
        return text.strip()
    except Exception as e:
        print(f"Video extraction error: {e}")
        return ""

# -------------------- DOCUMENT PROCESSING --------------------

def chunk_text(text, chunk_size=400):
    """Breaks large documents into smaller chunks."""
    sentences = text.split(". ")
    chunks = []
    current_chunk = []

    for sentence in sentences:
        current_chunk.append(sentence)
        if len(" ".join(current_chunk)) > chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = []

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def index_document(text):
    """Index document chunks in FAISS."""
    global documents, faiss_index
    
    if not retrieval_model:
        print("Retrieval model not loaded")
        return

    chunks = chunk_text(text)

    for chunk in chunks:
        documents.append(chunk)
        try:
            embedding = retrieval_model.encode([chunk])[0]
            faiss_index.add(np.array([embedding], dtype=np.float32))
        except Exception as e:
            print(f"Indexing error: {e}")

@app.route("/upload", methods=["POST"])
def upload_file():
    global uploaded_file_path  

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    upload_folder = "uploads"
    os.makedirs(upload_folder, exist_ok=True)

    uploaded_file_path = os.path.join(upload_folder, file.filename)
    file.save(uploaded_file_path)

    # Extract text from the file
    try:
        if file.filename.endswith(".pdf"):
            text = extract_text_from_pdf(uploaded_file_path)
        elif file.filename.endswith(".docx"):
            text = extract_text_from_docx(uploaded_file_path)
        elif file.filename.endswith(".csv"):
            text = extract_text_from_csv(uploaded_file_path)
        elif file.filename.endswith(('.png', '.jpg', '.jpeg')):
            text = extract_text_from_image(uploaded_file_path)
        elif file.filename.endswith(('.mp4', '.avi', '.mov')):
            text = extract_text_from_video(uploaded_file_path)
        else:
            return jsonify({"error": "Unsupported file format"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    if text:
        index_document(text)
        return jsonify({"message": f"File '{file.filename}' uploaded and indexed!"})
    return jsonify({"error": "No text extracted"}), 400

# -------------------- RETRIEVAL & RESPONSE GENERATION --------------------

def retrieve_relevant_text(query, top_k=3):
    """Hybrid Retrieval: Simple search."""
    if faiss_index.ntotal == 0 or not retrieval_model:
        return []

    try:
        # FAISS Search (Semantic Similarity)
        query_embedding = retrieval_model.encode([query])[0]
        _, indices = faiss_index.search(np.array([query_embedding], dtype=np.float32), top_k)
        faiss_results = [documents[idx] for idx in indices[0] if idx < len(documents)]
        
        return faiss_results[:top_k]
    except Exception as e:
        print(f"Retrieval error: {e}")
        return []

def generate_answer(context, query):
    """Generate answer using FLAN-T5."""
    if not t5_model or not t5_tokenizer:
        return "Model not loaded. Cannot generate answer."

    if not context.strip():
        return "I couldn't find relevant information."

    try:
        input_text = f"Given this document: {context} \nAnswer the question: {query}"
        input_ids = t5_tokenizer.encode(input_text, return_tensors="pt")
        output_ids = t5_model.generate(input_ids, max_length=300)

        return t5_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    except Exception as e:
        print(f"Answer generation error: {e}")
        return "Error generating answer."

@app.route('/answer', methods=['POST'])
def answer_query():
    global uploaded_file_path

    data = request.get_json()
    if not data:
        return jsonify({"answer": "No query received"}), 400
    
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"answer": "Invalid query"}), 400

    if not uploaded_file_path:
        return jsonify({"answer": "No file uploaded. Please upload a file first."}), 400

    relevant_texts = retrieve_relevant_text(query)
    combined_context = " ".join(relevant_texts)
    answer = generate_answer(combined_context, query)

    return jsonify({"answer": answer})

@app.route("/clear_faiss", methods=["POST"])
def clear_faiss():
    """Clear FAISS index and reset stored documents."""
    global documents, faiss_index
    documents = []
    faiss_index = faiss.IndexFlatL2(384)  
    return jsonify({"message": "FAISS index cleared!"})

if __name__ == "__main__":
    if not os.path.exists("uploads"):
        os.makedirs("uploads")
    app.run(debug=True)
