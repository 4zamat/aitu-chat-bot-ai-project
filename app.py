import streamlit as st
from alem_llm_handler import generate_llm_answer
from chatbot_logic_alem import load_precomputed_data, find_best_match_alem

# Cache models to prevent reloading on every interaction
@st.cache_resource
def load_all_models():
    """
    Loads precomputed Alem embeddings once.
    Cached by Streamlit to avoid reloading on every message.
    """
    print("--- ЗАГРУЗКА ДАННЫХ И ВЕКТОРОВ (ALEM) ---")
    texts, vectors, data = load_precomputed_data()
    
    if texts is None:
        st.error("Ошибка: не удалось загрузить файл alem_embeddings.pkl!")
        return None, None, None
    
    print("--- МОДЕЛИ ГОТОВЫ (ALEM) ---")
    return texts, vectors, data

# Load models once (cached)
precomputed_texts, precomputed_vectors, faq_data = load_all_models()

# Initialize UI and session state
st.title("🤖 FAQ-Бот AITU (Alem.ai RAG)")
st.caption("На базе Alem Embedder, Reranker и LLM")

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
        final_prompt = f"{prompt} {st.session_state.pending_topic}"
        st.session_state.pending_topic = None
    else:
        final_prompt = prompt

    # Note: Query expansion removed - Reranker handles relevance ranking
    
    # Plan A: Run Alem search pipeline
    print(f"Alem-Пайплайн: Ищу по запросу: '{final_prompt}'")
    
    found_contexts_list = find_best_match_alem(
        final_prompt, 
        precomputed_texts, 
        precomputed_vectors, 
        faq_data
    )
    
    # Generate response (Plan A: RAG or Plan C: Disambiguation)
    with st.chat_message("assistant"):
        with st.spinner("Думаю (Alem.ai)..."):
            
            if found_contexts_list:
                # Plan A: RAG - use original prompt for LLM
                print(f"ПЛАН А: Alem-Поиск нашел {len(found_contexts_list)} контекста. Запускаю RAG.")
                llm_answer = generate_llm_answer(prompt, found_contexts_list)
                st.markdown(llm_answer)
                
            else:
                # Plan C: Disambiguation - ask for clarification
                print("ПЛАН C: Alem-Поиск не нашел. Запрашиваю уточнение.")
                response_text = f"Я вижу, вас интересует тема: **'{prompt}'**. \n\n" \
                                "Не могли бы вы уточнить, что именно вы хотите узнать?"
                st.markdown(response_text)
                # Store topic in memory for next interaction
                st.session_state.pending_topic = prompt

    # Save assistant response to chat history
    if 'llm_answer' in locals():
        st.session_state.messages.append({"role": "assistant", "content": llm_answer})
    elif 'response_text' in locals():
        st.session_state.messages.append({"role": "assistant", "content": response_text})