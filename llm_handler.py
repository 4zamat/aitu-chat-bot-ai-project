import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

# Configure API key (prefer environment variable for production)
try:
    YOUR_API_KEY = os.getenv("GOOGLE_API_KEY")

    if YOUR_API_KEY is None:
        print("Ошибка: GOOGLE_API_KEY не найден. Создайте .env файл.")
        
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

def generate_llm_answer(user_question, found_contexts_list):
    """
    Generates answer using RAG approach with multiple contexts (k=3).
    Takes user question and list of retrieved contexts, sends to LLM for synthesis.
    """
    if YOUR_API_KEY is None:
        return "Ошибка: API-ключ для LLM не настроен."

    context_text = ""
    if not found_contexts_list:
        return "Ошибка: В LLM передан пустой список контекстов."

    # Format all contexts for prompt
    for i, context in enumerate(found_contexts_list):
        context_text += f"\n--- КОНТЕКСТ {i+1} ---\n"
        context_text += f"Вопрос из базы: {context.get('questions', '')}\n"
        context_text += f"Ответ из базы: {context.get('answers', '')}\n"
    
    # RAG prompt template with strict context-only rules
    prompt_template = f"""
    Ты — профессиональный, вежливый и дружелюбный ИИ-ассистент 
    Astana IT University (AITU).

    ТВОЯ ЗАДАЧА:
    Внимательно изучи "ВОПРОС ПОЛЬЗОВАТЕЛЯ" и предоставленные "КОНТЕКСТЫ".
    Твой ответ должен быть синтезом (объединением) информации 
    из ВСЕХ релевантных контекстов.

    ПРАВИЛА ОТВЕТА:
    1.  **ТОЛЬКО КОНТЕКСТ:** Отвечай, используя СТРОГО 
        информацию из предоставленных "КОНТЕКСТОВ". 
        Не придумывай ничего от себя.
    2.  **НЕРЕЛЕВАНТНЫЙ КОНТЕКСТ (ГЛАВНОЕ ПРАВИЛО):** Если ни один из "КОНТЕКСТОВ" НЕ содержит прямого 
        ответа на "ВОПРОС ПОЛЬЗОВАТЕЛЯ", не пытайся 
        выдумать ответ. Вместо этого вежливо скажи: 
        "Я нашел информацию по этой теме, но, к сожалению, 
        не точный ответ на ваш вопрос. Не могли бы вы 
        его переформулировать?"
    3.  **ТОН:** Твой тон должен быть официальным, но 
        приветливым. Всегда обращайся к пользователю на "Вы".
    4.  **СИНТЕЗ:** Если несколько контекстов говорят об 
        одном и том же, объедини их в один плавный ответ.

    ---
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
        return f"(API Ошибка) Вот что я нашел: {found_contexts_list[0].get('answers')}"

def generate_fallback_answer(user_question):
    """
    Fallback function when SBERT retrieval finds no relevant context.
    Sends question to LLM without RAG context, using general knowledge.
    """
    if YOUR_API_KEY is None:
        return "Ошибка: API-ключ для LLM не настроен."

    # Fallback prompt - instructs LLM to use general knowledge with disclaimer
    prompt_template = f"""
    Ты — полезный ИИ-ассистент Astana IT University (AITU).
    
    ВАЖНО: Поиск по базе знаний AITU не дал результатов 
    по следующему вопросу пользователя.

    ТВОЯ ЗАДАЧА:
    1.  Попробуй вежливо ответить на "ВОПРОС ПОЛЬЗОВАТЕЛЯ", 
        используя свои **общие знания**.
    2.  **ОБЯЗАТЕЛЬНО** начни свой ответ с вежливого 
        предупреждения, что ты не нашел эту информацию в 
        базе знаний AITU и отвечаешь на основе общих знаний.
    
    Пример начала ответа:
    "Я не смог найти точную информацию об этом в базе 
    знаний AITU, но, основываясь на общих данных, могу 
    сказать, что..."

    ---
    ВОПРОС ПОЛЬЗОВАТЕЛЯ:
    {user_question}
    ---
    
    ОТВЕТ АССИСТЕНТА (С ПРЕДУПРЕЖДЕНИЕМ):
    """
    
    try:
        response = model.generate_content(prompt_template)
        return response.text
    except Exception as e:
        print(f"Ошибка при вызове LLM API (Fallback): {e}")
        return "Извините, я не смог найти информацию в базе знаний, и мой резервный помощник тоже не отвечает."