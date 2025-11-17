import streamlit as st
from data_handler import load_data_from_csv
from chatbot_logic_v3 import create_sbert_embeddings, find_best_match_sbert
from llm_handler import generate_llm_answer

# Cache models to prevent reloading on every interaction
# @st.cache_resource ensures function runs once and caches result in memory
@st.cache_resource
def load_all_models():
    """
    Loads SBERT model, data, and creates embeddings once.
    Cached by Streamlit to avoid reloading on every message.
    """
    print("--- ЗАГРУЗКА МОДЕЛЕЙ (ВЫПОЛНЯЕТСЯ ОДИН РАЗ) ---")
    DATA_FILE = "data/QA_cleaned.csv" 
    faq_data = load_data_from_csv(DATA_FILE)
    
    if faq_data is None:
        st.error("Ошибка: не удалось загрузить файл с данными.")
        return None, None

    print("Создаю объединенный текстовый корпус для SBERT...")
    combined_texts = [f"Вопрос: {item.get('questions', '')} Ответ: {item.get('answers', '')}" for item in faq_data]
    
    context_embeddings = create_sbert_embeddings(combined_texts)
    
    if context_embeddings is None:
        st.error("Ошибка: не удалось создать SBERT векторы.")
        return None, None
    
    print("--- МОДЕЛИ ГОТОВЫ ---")
    return faq_data, context_embeddings

# Load models once (cached)
faq_data, context_embeddings = load_all_models()

# Initialize UI and session state
st.title("🤖 FAQ-Бот AITU (RAG v3.4)")
st.caption("На базе SBERT, Gemini и Streamlit")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    
# Initialize pending topic for disambiguation (Plan C)
if "pending_topic" not in st.session_state:
    st.session_state.pending_topic = None

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Main chat input handler
if prompt := st.chat_input("Спросите что-нибудь о AITU..."):
    
    # Add and display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    # Plan C: Check if we're waiting for clarification
    if st.session_state.pending_topic:
        # Combine clarification with pending topic
        print(f"ПЛАН C: Объединяю '{prompt}' с темой '{st.session_state.pending_topic}'")
        final_prompt = f"{prompt} {st.session_state.pending_topic}"
        st.session_state.pending_topic = None
    else:
        final_prompt = prompt

    # Query expansion for short queries (< 3 words)
    if len(final_prompt.split()) < 3:
        print("... (Query Expansion)...")
        sbert_query = f"Вопрос по теме: {final_prompt}"
    else:
        sbert_query = final_prompt
    
    # Plan A: Run SBERT search
    print(f"SBERT Ищет по запросу: '{sbert_query}'")
    found_contexts_list = find_best_match_sbert(sbert_query, 
                                                context_embeddings, 
                                                faq_data)
    
    # Generate response (Plan A: RAG or Plan C: Disambiguation)
    with st.chat_message("assistant"):
        with st.spinner("Думаю..."):
            if found_contexts_list:
                # Plan A: RAG - use original (non-expanded) prompt for LLM
                print("ПЛАН А: SBERT нашел контекст. Запускаю RAG.")
                llm_answer = generate_llm_answer(prompt, found_contexts_list)
                st.markdown(llm_answer)
                
            else:
                # Plan C: Disambiguation - ask for clarification
                print("ПЛАН C: SBERT не нашел. Запрашиваю уточнение.")
                
                response_text = f"Я вижу, вас интересует тема: **'{prompt}'**. \n\n" \
                                "Не могли бы вы уточнить, что именно вы хотите узнать? \n\n" \
                                "(Например: *стоимость*, *расположение*, *документы* и т.д.)"
                
                st.markdown(response_text)
                
                # Store topic in memory for next interaction
                st.session_state.pending_topic = prompt

    # Save assistant response to chat history
    if 'llm_answer' in locals():
        st.session_state.messages.append({"role": "assistant", "content": llm_answer})
    elif 'response_text' in locals():
        st.session_state.messages.append({"role": "assistant", "content": response_text})   