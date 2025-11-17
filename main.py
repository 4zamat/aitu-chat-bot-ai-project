from data_handler import load_data_from_csv
from chatbot_logic_v3 import create_sbert_embeddings, find_best_match_sbert

def main():
    """
    Main function for chatbot startup (Version 3.0 with SBERT).
    """
    DATA_FILE = "QA_addmissionAitu.csv" 
    faq_data = load_data_from_csv(DATA_FILE)
    
    if faq_data is None:
        return 

    # Extract original (unprocessed) questions - SBERT works better with full text
    questions_from_db = []
    for item in faq_data:
        questions_from_db.append(item.get('questions', ''))
    
    # Create embeddings once at startup
    question_embeddings = create_sbert_embeddings(questions_from_db)
    
    print("\nМодель SBERT готова. Запуск чат-бота...")
    while True:
        print("\n=== Меню FAQ-Бота AITU (v3.0 SBERT) ===")
        print("1. Задать вопрос")
        print("2. Выход")
        
        choice = input("Выберите опцию (1-2): ")
        
        if choice == '1':
            user_question = input("Введите ваш вопрос: ")
            
            answer = find_best_match_sbert(user_question, 
                                           question_embeddings, 
                                           faq_data)
            
            print("\nОтвет Бота:")
            print(answer)
            
        elif choice == '2':
            print("До свидания!")
            break
            
        else:
            print("Некорректный ввод. Пожалуйста, выберите 1 или 2.")

if __name__ == "__main__":
    main()