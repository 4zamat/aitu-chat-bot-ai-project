from data_handler import load_data_from_csv, preprocess_text
from chatbot_logic import fit_tfidf_vectorizer, find_best_match

def main():
    DATA_FILE = "QA_addmissionAitu.csv" 
    
    faq_data = load_data_from_csv(DATA_FILE)
    
    if faq_data is None:
        return 

    print("Начинаю предварительную обработку текста...")
    cleaned_questions = []
    for item in faq_data:
        cleaned_q = preprocess_text(item.get('questions', ''))
        cleaned_questions.append(cleaned_q)
    
    # Train TF-IDF model once at startup
    vectorizer, tfidf_matrix = fit_tfidf_vectorizer(cleaned_questions)
    
    print("\nМодель готова. Запуск чат-бота...")
    
    while True:
        print("\n=== Меню FAQ-Бота AITU ===")
        print("1. Задать вопрос")
        print("2. Выход")
        
        choice = input("Выберите опцию (1-2): ")
        
        if choice == '1':
            user_question = input("Введите ваш вопрос: ")
            
            answer = find_best_match(user_question, vectorizer, tfidf_matrix, faq_data)
            print("\nОтвет Бота:")
            print(answer)
            
        elif choice == '2':
            print("До свидания!")
            break
            
        else:
            print("Некорректный ввод. Пожалуйста, выберите 1 или 2.")

if __name__ == "__main__":
    main()