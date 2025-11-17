from sentence_transformers import SentenceTransformer
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# Load SBERT model once at import time
MODEL_NAME = 'distiluse-base-multilingual-cased-v1'
print(f"Загрузка SBERT модели '{MODEL_NAME}'...")
sbert_model = SentenceTransformer(MODEL_NAME)
print("SBERT модель успешно загружена.")

def create_sbert_embeddings(questions):
    """Creates SBERT embeddings (vector representations) for a list of questions."""
    print("Создаю SBERT-векторы для базы вопросов...")
    question_embeddings = sbert_model.encode(questions, 
                                           show_progress_bar=True,
                                           convert_to_tensor=True)
    return question_embeddings

def prepare_training_data(data_list):
    """
    Prepares training data:
    - Creates lookup dictionary mapping class labels to canonical answers
    - Extracts questions (X) and labels (Y)
    - Encodes labels to numeric format required by ML models
    """
    # Use first answer found for each class as canonical answer
    answer_lookup = {}
    
    questions = []
    labels = []
    
    for item in data_list:
        label = item.get('classes', 'unknown')
        question = item.get('questions', '')
        answer = item.get('answers', 'Нет ответа для этого класса.')
        
        questions.append(question)
        labels.append(label)
        
        if label not in answer_lookup:
            answer_lookup[label] = answer
            
    # Encode string labels to numeric format: ['AITU_overview', 'AITU_admission'] -> [0, 1]
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    
    print("Данные для обучения подготовлены.")
    return questions, labels_encoded, label_encoder, answer_lookup

def train_classifier(X_features, y_labels):
    """Trains a classifier on SBERT embeddings (X) and class labels (y)."""
    print("Обучаю классификатор LogisticRegression...")
    
    # Use 'ovr' (one-vs-rest) for multi-class classification
    classifier = LogisticRegression(multi_class='ovr', max_iter=200)
    
    # Convert tensors from GPU to CPU for scikit-learn compatibility
    classifier.fit(X_features.cpu().numpy(), y_labels)
    
    print("Классификатор обучен.")
    return classifier

def predict_intent(user_question, classifier, label_encoder, answer_lookup):
    """
    Predicts the intent (class) of user's question and returns corresponding answer.
    """
    # Encode user question to SBERT embedding
    user_embedding = sbert_model.encode(user_question, convert_to_tensor=True)
    # Reshape for scikit-learn: (embedding_dim,) -> (1, embedding_dim)
    user_vector = user_embedding.cpu().numpy().reshape(1, -1)
    
    # Predict class
    predicted_class_encoded = classifier.predict(user_vector)
    
    # Decode numeric class back to label string
    predicted_class_label = label_encoder.inverse_transform(predicted_class_encoded)[0]
    
    # Retrieve answer from lookup dictionary
    answer = answer_lookup.get(predicted_class_label, "Не нашел ответ, хотя класс распознал.")
    
    return answer, predicted_class_label