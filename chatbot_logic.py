from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from data_handler import preprocess_text

def fit_tfidf_vectorizer(cleaned_questions):
    """Trains TF-IDF vectorizer and returns vectorizer and matrix."""
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(cleaned_questions)
    
    print("TF-IDF модель обучена.")
    return vectorizer, tfidf_matrix

def find_best_match(user_question, vectorizer, tfidf_matrix, data_list):
    """Finds the most similar question using cosine similarity."""
    cleaned_user_question = preprocess_text(user_question)
    
    # Use transform(), not fit_transform() - vectorizer is already trained
    user_vector = vectorizer.transform([cleaned_user_question])
    similarities = cosine_similarity(user_vector, tfidf_matrix)
    
    best_match_index = similarities[0].argmax()
    best_score = similarities[0][best_match_index]
    
    SIMILARITY_THRESHOLD = 0.2
    
    if best_score > SIMILARITY_THRESHOLD:
        return data_list[best_match_index].get('answers', 'Ответ не найден.')
    else:
        return "Извините, я не смог найти подходящий ответ на ваш вопрос."