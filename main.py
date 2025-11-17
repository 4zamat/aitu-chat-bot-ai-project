from data_handler import load_data_from_csv
from chatbot_logic_v3 import create_sbert_embeddings, find_best_match_sbert
from llm_handler import generate_llm_answer

def main():
    """
    Main function for chatbot startup (Version 3 + RAG).
    Implements RAG pipeline: Retrieval (SBERT) -> Generation (LLM).
    """
    DATA_FILE = "QA_addmissionAitu.csv" 
    faq_data = load_data_from_csv(DATA_FILE)
    if faq_data is None: return 

    questions_from_db = [item.get('questions', '') for item in faq_data]
    
    question_embeddings = create_sbert_embeddings(questions_from_db)
    if question_embeddings is None: return

    print("\nМодель SBERT + RAG готова. Запуск чат-бота...")
    while True:
        print("\n=== Меню FAQ-Бота AITU (v3 + RAG) ===")
        print("1. Задать вопрос")
        print("2. Выход")
        
        choice = input("Выберите опцию (1-2): ")
        
        if choice == '1':
            user_question = input("Введите ваш вопрос: ")
            
            # Step 1: Retrieval - find relevant context using SBERT
            found_context = find_best_match_sbert(user_question, 
                                                  question_embeddings, 
                                                  faq_data)
            
            print("\nОтвет Бота (RAG):")
            
            if found_context:
                # Step 2: Generation - send question and context to LLM
                print("... (SBERT нашел контекст. Отправляю в LLM)...")
                
                llm_answer = generate_llm_answer(user_question, found_context)
                
                print(llm_answer)
            else:
                # No relevant context found (similarity score < threshold)
                print("Извините, я не смог найти релевантную информацию по вашему вопросу.")
            
        elif choice == '2':
            print("До свидания!")
            break
            
        else:
            print("Некорректный ввод. Пожалуйста, выберите 1 или 2.")

if __name__ == "__main__":
    main()