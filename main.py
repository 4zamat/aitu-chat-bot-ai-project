from data_handler import load_data_from_csv
from chatbot_logic_v3 import create_sbert_embeddings, find_best_match_sbert
from llm_handler import generate_llm_answer, generate_fallback_answer

def main():
    """
    Main function for chatbot startup (Version 3 + RAG).
    Implements RAG pipeline: Retrieval (SBERT) -> Generation (LLM).
    """
    DATA_FILE = "data/QA_addmissionAitu.csv" 
    faq_data = load_data_from_csv(DATA_FILE)
    if faq_data is None: return 

    print("Создаю объединенный текстовый корпус для SBERT...")
    # Combine question and answer into single document for better retrieval
    combined_texts = []
    for item in faq_data:
        q = item.get('questions', '')
        a = item.get('answers', '')
        combined_texts.append(f"Вопрос: {q} Ответ: {a}")

    # Create embeddings from combined texts
    question_embeddings = create_sbert_embeddings(combined_texts)
    if question_embeddings is None: return

    print("\nМодель SBERT + RAG готова. Запуск чат-бота...")
    while True:
        print("\n=== Меню FAQ-Бота AITU (v3 + RAG) ===")
        print("1. Задать вопрос")
        print("2. Выход")
        
        choice = input("Выберите опцию (1-2): ")
        
        if choice == '1':
            user_question = input("Введите ваш вопрос: ")
            
            # Query expansion for short queries (< 3 words)
            if len(user_question.split()) < 3:
                print("... (Короткий запрос. Активирую Query Expansion)...")
                user_question_expanded = f"Вопрос по теме: {user_question}"
            else:
                user_question_expanded = user_question
                
            # Step 1: Retrieval - find relevant context using SBERT
            found_contexts_list = find_best_match_sbert(user_question, 
                                                  question_embeddings, 
                                                  faq_data)
            
            print("\nОтвет Бота (RAG):")
            
            if found_contexts_list:
                # Plan A: RAG - retrieval successful
                print(f"... (SBERT нашел {len(found_contexts_list)} контекста(ов). Отправляю в LLM)...")
                llm_answer = generate_llm_answer(user_question, found_contexts_list)
                print(llm_answer)
            else:
                # Plan B: Fallback - retrieval failed, use LLM without context
                print("... (SBERT ничего не нашел. Активирую Fallback-логику)...")
                llm_answer = generate_fallback_answer(user_question)
                print(llm_answer)
            
        elif choice == '2':
            print("До свидания!")
            break
            
        else:
            print("Некорректный ввод. Пожалуйста, выберите 1 или 2.")

if __name__ == "__main__":
    main()