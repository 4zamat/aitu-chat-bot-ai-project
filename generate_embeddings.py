import os
import requests
import time
import pickle
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv

from data_handler import load_data_from_csv

# Load embedder API key from .env
load_dotenv()
EMBED_API_KEY = os.getenv("EMBED_API_KEY")

if not EMBED_API_KEY:
    raise ValueError("EMBED_API_KEY не найден в вашем .env файле!")

# Configuration
EMBEDDER_URL = "https://llm.alem.ai/v1/embeddings"
EMBEDDER_MODEL = "text-1024"
OUTPUT_FILE = "alem_embeddings.pkl"

EMBEDDER_HEADERS = {
    "Authorization": f"Bearer {EMBED_API_KEY}",
    "Content-Type": "application/json"
}

def get_embedding(text):
    """
    Calls Alem Embedder API to get embedding for a single text.
    """
    payload = {
        "model": EMBEDDER_MODEL,
        "input": text
    }

    try:
        response = requests.post(EMBEDDER_URL, json=payload, headers=EMBEDDER_HEADERS, timeout=60)
        response.raise_for_status()
        data = response.json()
        embedding = data["data"][0]["embedding"]
        return embedding
    except Exception as e:
        print(f"Ошибка при получении вектора для текста: '{text[:20]}...': {e}")
        return None

def main_generate():
    """
    Main function to generate embeddings for all FAQ documents.
    Combines questions and answers, gets embeddings via API, and saves to pickle file.
    """
    print("--- Запуск скрипта генерации векторов (Alem Embedder) ---")
    
    DATA_FILE = "data/QA_addmissionAitu.csv" 
    faq_data = load_data_from_csv(DATA_FILE)
    if not faq_data:
        print("Ошибка: не удалось загрузить данные.")
        return

    print("Создаю объединенный текстовый корпус (Q+A)...")
    # Combine question and answer into single document for better retrieval
    combined_texts = []
    for item in faq_data:
        q = item.get('questions', '')
        a = item.get('answers', '')
        combined_texts.append(f"Вопрос: {q} Ответ: {a}")

    print(f"Получаю векторы для {len(combined_texts)} документов. Это может занять время...")
    
    all_vectors = []
    texts_with_vectors = []
    
    for text in tqdm(combined_texts):
        vector = get_embedding(text)
        if vector:
            all_vectors.append(vector)
            texts_with_vectors.append(text)
        
        # Rate limiting for API
        time.sleep(1.1) 

    if not all_vectors:
        print("Ошибка: не получено ни одного вектора.")
        return

    embeddings_matrix = np.array(all_vectors)

    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump({
            "texts": texts_with_vectors,
            "vectors": embeddings_matrix,
            "original_data": faq_data
        }, f)
        
    print(f"\nГотово! Векторы ({embeddings_matrix.shape}) сохранены в '{OUTPUT_FILE}'")

if __name__ == "__main__":
    main_generate()