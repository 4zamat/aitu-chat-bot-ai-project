from sentence_transformers import SentenceTransformer, util
import torch 

# 1. Загрузка SBERT модели
MODEL_NAME = 'distiluse-base-multilingual-cased-v1'
print(f"Загрузка SBERT модели '{MODEL_NAME}'... Это может занять время.")
try:
    sbert_model = SentenceTransformer(MODEL_NAME)
    print("SBERT модель успешно загружена.")
except Exception as e:
    print(f"Ошибка при загрузке модели: {e}")
    sbert_model = None

def create_sbert_embeddings(questions):
    """
    Создает SBERT-векторы (embeddings) для списка вопросов.
    """
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
    Находит лучший ответ, используя SBERT и косинусное сходство.
    (Это и есть "Retrieval" компонент для RAG)
    """
    if sbert_model is None:
        return "Ошибка: SBERT модель не загружена."

    # 1. Кодируем вопрос пользователя
    user_embedding = sbert_model.encode(user_question, convert_to_tensor=True)
    
    # 2. Считаем косинусное сходство
    cos_scores = util.cos_sim(user_embedding, question_embeddings)[0]
    
    # 3. Находим лучший результат
    top_results = torch.topk(cos_scores, k=1)
    
    best_score = top_results[0].item()
    best_match_index = top_results[1].item()
    
    # 4. Проверяем порог
    SIMILARITY_THRESHOLD = 0.5 # Можно поиграть с этим значением
    
    if best_score > SIMILARITY_THRESHOLD:
        # ВАЖНО: Мы возвращаем НЕ СРАЗУ ОТВЕТ, 
        # а ВЕСЬ найденный 'item' (вопрос + ответ).
        # Это будет наш КОНТЕКСТ для LLM.
        found_item = data_list[best_match_index]
        return found_item
    else:
        return None # Возвращаем None, если ничего не найдено