import os
import requests
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

# Load API keys from .env
load_dotenv()
EMBED_API_KEY = os.getenv("EMBED_API_KEY")
RERANK_API_KEY = os.getenv("RERANK_API_KEY")

if not EMBED_API_KEY or not RERANK_API_KEY:
    raise ValueError("EMBED_API_KEY или RERANK_API_KEY не найдены в .env файле!")

# Configuration
EMBEDDINGS_FILE = "alem_embeddings.pkl"
TOP_K_RETRIEVAL = 20  # Number of candidates for coarse search
TOP_N_RERANK = 3  # Number of final results after reranking

# Embedder API configuration
EMBEDDER_URL = "https://llm.alem.ai/v1/embeddings"
EMBEDDER_MODEL = "text-1024"
EMBEDDER_HEADERS = {
    "Authorization": f"Bearer {EMBED_API_KEY}",
    "Content-Type": "application/json"
}

# Reranker API configuration
RERANKER_URL = "https://reranker-llm.alem.ai/v1/rerank"
RERANKER_HEADERS = {
    "Authorization": f"Bearer {RERANK_API_KEY}",
    "Content-Type": "application/json"
}

def load_precomputed_data():
    """
    Loads precomputed embeddings and original data from pickle file.
    Note: Caching is handled in app.py using @st.cache_resource.
    """
    print("Загружаю пред-рассчитанные векторы (Alem)...")
    try:
        with open(EMBEDDINGS_FILE, "rb") as f:
            data = pickle.load(f)
        return data["texts"], data["vectors"], data["original_data"]
    except FileNotFoundError:
        print(f"ОШИБКА: Файл {EMBEDDINGS_FILE} не найден!")
        print("Пожалуйста, сначала запустите generate_embeddings.py")
        return None, None, None

def get_embedding_for_query(text):
    """
    Calls Alem Embedder API to get embedding for a single user query.
    """
    payload = {"model": EMBEDDER_MODEL, "input": text}
    try:
        response = requests.post(EMBEDDER_URL, json=payload, headers=EMBEDDER_HEADERS, timeout=60)
        response.raise_for_status()
        data = response.json()
        return data["data"][0]["embedding"]
    except Exception as e:
        print(f"Ошибка при получении вектора для запроса: {e}")
        return None

def rerank_documents(query, documents_list):
    """
    Calls Alem Reranker API to rerank list of documents based on query relevance.
    """
    payload = {
        "query": query,
        "documents": documents_list,
        "top_n": TOP_N_RERANK
    }
    
    try:
        response = requests.post(RERANKER_URL, json=payload, headers=RERANKER_HEADERS, timeout=60)
        response.raise_for_status()
        data = response.json()
        
        # Parse Reranker response
        reranked_texts = [result["document"]["text"] for result in data.get("results", [])]
        return reranked_texts

    except Exception as e:
        print(f"Ошибка при вызове Reranker API: {e}")
        return []

def find_best_match_alem(query, precomputed_texts, precomputed_vectors, original_data):
    """
    Full retrieval pipeline: Embed Query -> Coarse Search (k=20) -> Rerank (k=3).
    Returns list of context dictionaries for RAG.
    """
    
    # Step 1: Embed user query
    print(f"Alem-Поиск: Векторизую запрос '{query}'...")
    query_vector = get_embedding_for_query(query)
    if query_vector is None:
        return None

    query_vector = np.array(query_vector).reshape(1, -1)
    
    # Step 2: Coarse search using cosine similarity (local computation)
    print(f"Alem-Поиск: Ищу {TOP_K_RETRIEVAL} кандидатов (Coarse Search)...")
    scores = cosine_similarity(query_vector, precomputed_vectors)
    
    top_k_indices = np.argsort(scores[0])[-TOP_K_RETRIEVAL:][::-1]
    
    candidate_texts_for_reranker = [precomputed_texts[i] for i in top_k_indices]
    
    # Step 3: Fine search using Reranker API
    print(f"Alem-Поиск: Отправляю {len(candidate_texts_for_reranker)} кандидатов в Reranker...")
    reranked_texts = rerank_documents(query, candidate_texts_for_reranker)
    
    if not reranked_texts:
        print("Alem-Поиск: Reranker ничего не вернул.")
        return None
    
    print(f"Alem-Поиск: Reranker вернул {len(reranked_texts)} лучших.")

    # Step 4: Map reranked texts back to original data dictionaries
    text_to_dict_map = {precomputed_texts[i]: original_data[i] for i in range(len(original_data))}
    
    final_contexts_list = []
    for text in reranked_texts:
        if text in text_to_dict_map:
            final_contexts_list.append(text_to_dict_map[text])
        
    return final_contexts_list