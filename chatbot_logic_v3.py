from sentence_transformers import SentenceTransformer, util
import torch

# Load multilingual SBERT model once at import time
# This model handles Russian text well without preprocessing
MODEL_NAME = 'distiluse-base-multilingual-cased-v1'
print(f"Загрузка SBERT модели '{MODEL_NAME}'... Это может занять время.")
model = SentenceTransformer(MODEL_NAME)
print("SBERT модель успешно загружена.")


def create_sbert_embeddings(questions):
    """
    Creates SBERT embeddings (vector representations) for a list of questions.
    """
    print("Создаю SBERT-векторы для базы вопросов...")
    question_embeddings = model.encode(questions, 
                                       show_progress_bar=True, 
                                       convert_to_tensor=True)
    print("Векторы (embeddings) созданы.")
    return question_embeddings


def find_best_match_sbert(user_question, question_embeddings, data_list):
    """
    Finds the best matching answer using SBERT embeddings and cosine similarity.
    Note: SBERT handles punctuation and case automatically, no preprocessing needed.
    """
    # Encode user question into embedding vector
    user_embedding = model.encode(user_question, convert_to_tensor=True)
    
    # Compute cosine similarity between user question and all database questions
    cos_scores = util.cos_sim(user_embedding, question_embeddings)[0]
    
    # Find top match using torch.topk
    top_results = torch.topk(cos_scores, k=1)
    
    best_score = top_results[0].item()
    best_match_index = top_results[1].item()
    
    # SBERT typically requires higher threshold (0.4-0.6) compared to TF-IDF
    SIMILARITY_THRESHOLD = 0.5
    
    if best_score > SIMILARITY_THRESHOLD:
        return data_list[best_match_index].get('answers', 'Ответ не найден.')
    else:
        return "Извините, я не смог найти подходящий ответ на ваш вопрос (SBERT)."