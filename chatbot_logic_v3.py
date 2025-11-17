from sentence_transformers import SentenceTransformer, util
import torch 

# Load multilingual SBERT model once at import time
MODEL_NAME = 'distiluse-base-multilingual-cased-v1'
print(f"Загрузка SBERT модели '{MODEL_NAME}'... Это может занять время.")
try:
    sbert_model = SentenceTransformer(MODEL_NAME)
    print("SBERT модель успешно загружена.")
except Exception as e:
    print(f"Ошибка при загрузке модели: {e}")
    sbert_model = None

def create_sbert_embeddings(questions):
    """Creates SBERT embeddings (vector representations) for a list of questions."""
    if sbert_model is None:
        print("Ошибка: SBERT модель не была загружена.")
        return None
        
    print("Создаю SBERT-векторы для базы вопросов...")
    question_embeddings = sbert_model.encode(questions, 
                                           show_progress_bar=True, 
                                           convert_to_tensor=True)
    print("Векторы (embeddings) созданы.")
    return question_embeddings


def find_best_match_sbert(user_question, question_embeddings, data_list):
    """
    Finds the best matching context using SBERT embeddings and cosine similarity.
    This is the Retrieval component for RAG - returns full context item, not just answer.
    """
    if sbert_model is None:
        return "Ошибка: SBERT модель не загружена."

    # Encode user question into embedding vector
    user_embedding = sbert_model.encode(user_question, convert_to_tensor=True)
    
    # Compute cosine similarity between user question and all database questions
    cos_scores = util.cos_sim(user_embedding, question_embeddings)[0]
    
    # Find top match using torch.topk
    K_VALUE = 3 
    top_results = torch.topk(cos_scores, k=K_VALUE)
    
    best_scores = top_results[0]
    best_indices = top_results[1]
    
    SIMILARITY_THRESHOLD = 0.5 
    
    found_items = []
    for i in range(K_VALUE):
        score = best_scores[i].item()
        index = best_indices[i].item()
        
        if score > SIMILARITY_THRESHOLD:
            found_items.append(data_list[index])
            
    if found_items:
        return found_items  # Return list of context dictionaries
    else:
        return None