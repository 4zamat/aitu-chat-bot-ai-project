import google.generativeai as genai
import os

# Configure API key (prefer environment variable for production)
try:
    YOUR_API_KEY = "AIzaSyBdMAgEPfsmDS8LApOgsnbPXl5bhsty9vU"
    genai.configure(api_key=YOUR_API_KEY)
except Exception as e:
    print(f"Ошибка конфигурации API: {e}")
    YOUR_API_KEY = None

# Configure Gemini model with generation parameters
generation_config = {
    "temperature": 0.7,  # Controls creativity/randomness of responses
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": 2048,
}

model = genai.GenerativeModel(model_name="gemini-2.0-flash", 
                              generation_config=generation_config)

def generate_llm_answer(user_question, found_context):
    """
    Generates answer using LLM with RAG approach.
    Takes user question and retrieved context, sends to LLM for generation.
    """
    if YOUR_API_KEY is None:
        return "Ошибка: API-ключ для LLM не настроен."

    context_text = found_context.get('answers', '')
    
    # RAG prompt template - instructs LLM to use only provided context
    prompt_template = f"""
    Ты — полезный ИИ-ассистент Astana IT University (AITU).
    Твоя задача — вежливо и кратко ответить на вопрос пользователя.
    
    Отвечай, используя ТОЛЬКО следующий Контекст.
    Не придумывай ничего от себя.
    Если Контекст нерелевантен, просто скажи, что не нашел ответа.
    
    ---
    КОНТЕКСТ:
    {context_text}
    ---
    
    ВОПРОС ПОЛЬЗОВАТЕЛЯ:
    {user_question}
    ---
    
    ОТВЕТ АССИСТЕНТА:
    """
    
    try:
        response = model.generate_content(prompt_template)
        return response.text
    except Exception as e:
        print(f"Ошибка при вызове LLM API: {e}")
        # Fallback to raw context if API call fails
        return f"(API Ошибка) Вот что я нашел: {context_text}"