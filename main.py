import cv2
import whisper
import pyttsx3
from ultralytics import YOLO
from deepface import DeepFace
from transformers import pipeline, AutoTokenizer, AutoModel
import torch
from google.colab.patches import cv2_imshow
import faiss
import pickle
import numpy as np

# Install dependencies

from sentence_transformers import SentenceTransformer

# Load Models
human_detector = YOLO("yolov8n.pt")
chatbot = pipeline("text-generation", model="gpt2", device=0 if torch.cuda.is_available() else -1)
tts_engine = pyttsx3.init()
whisper_model = whisper.load_model("base")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # Embedding model for RAG

# Load or create FAISS index for RAG
def load_faiss_index():
    try:
        with open("rag_index.pkl", "rb") as f:
            index, documents = pickle.load(f)
    except FileNotFoundError:
        index = faiss.IndexFlatL2(384)  # Embedding dimension
        documents = []
    return index, documents

rag_index, rag_documents = load_faiss_index()

# Function to add document to FAISS index
def add_document_to_rag(document_text):
    global rag_index, rag_documents
    embedding = embedding_model.encode([document_text])
    rag_index.add(np.array(embedding, dtype=np.float32))
    rag_documents.append(document_text)
    with open("rag_index.pkl", "wb") as f:
        pickle.dump((rag_index, rag_documents), f)

# Define policy rules
def is_question_allowed(question):
    """Check if the question adheres to policy rules."""
    restricted_topics = {"violence", "hate speech", "illegal activity"}
    return not any(topic in question.lower() for topic in restricted_topics)

def validate_response(response):
    """Ensure response follows educational guidelines and is factually correct."""
    restricted_words = {"fake", "incorrect", "misleading"}
    return "I'm sorry, but I can't provide an answer to that." if any(word in response.lower() for word in restricted_words) else response

def detect_human(frame):
    """Detect humans in a frame using YOLOv8."""
    results = human_detector(frame)
    return next(((int(x1), int(y1), int(x2), int(y2)) for x1, y1, x2, y2, conf, cls in results.xyxy[0] if int(cls) == 0), None)

def recognize_face(frame):
    """Recognize if the detected human is the correct one."""
    try:
        result = DeepFace.find(frame, db_path="faces_db/")
        return result["identity"][0] if result else None
    except:
        return None

def speech_to_text(audio_path):
    """Convert speech to text using Whisper."""
    return whisper_model.transcribe(audio_path)["text"]

def retrieve_rag_response(question):
    """Retrieve the best matching response from FAISS index."""
    if rag_index.ntotal == 0:
        return "No relevant document found."
    
    question_embedding = embedding_model.encode([question])
    _, indices = rag_index.search(np.array(question_embedding, dtype=np.float32), 1)
    return rag_documents[indices[0][0]] if indices[0][0] < len(rag_documents) else "No relevant document found."

def chatbot_response(question):
    """Generate a chatbot response using GPT-2 with policy validation and RAG."""
    if not is_question_allowed(question):
        return "I'm sorry, but I cannot answer that question."
    
    rag_answer = retrieve_rag_response(question)
    generated_response = chatbot(question, max_length=50)[0]["generated_text"]
    return validate_response(rag_answer + " " + generated_response)

def text_to_speech(text):
    """Convert text to speech using pyttsx3."""
    tts_engine.say(text)
    tts_engine.runAndWait()

def main():
    """Main function to run human detection and chatbot interaction."""
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue
        
        detected = detect_human(frame)
        if detected:
            x1, y1, x2, y2 = detected
            identity = recognize_face(frame[y1:y2, x1:x2])
            
            if identity:
                print("Correct human detected!")
                question = speech_to_text("input.wav")  # Replace with actual audio input
                answer = chatbot_response(question)
                print("Bot Response:", answer)
                text_to_speech(answer)
        
        cv2_imshow(frame)  # Use cv2_imshow for Colab compatibility
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
