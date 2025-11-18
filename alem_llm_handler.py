import os
import requests
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()
ALEM_API_KEY = os.getenv("ALEM_API_KEY")

if not ALEM_API_KEY:
    raise ValueError("ALEM_API_KEY не найден в вашем .env файле!")

ALEM_LLM_URL = "https://llm.alem.ai/v1/chat/completions"

HEADERS = {
    "Authorization": f"Bearer {ALEM_API_KEY}",
    "Content-Type": "application/json"
}

def _call_alem_api(prompt_string):
    """
    Internal function to call AlemLLM API.
    Uses OpenAI-compatible payload structure.
    """
    if not ALEM_API_KEY:
        return "Ошибка: ALEM_API_KEY не найден."

    # Wrap prompt string in OpenAI-compatible message format
    payload = {
        "model": "alemllm",
        "messages": [
            {
                "role": "user",
                "content": prompt_string
            }
        ]
    }

    try:
        response = requests.post(ALEM_LLM_URL, json=payload, headers=HEADERS, timeout=60)
        response.raise_for_status()  # Check for HTTP errors (4xx, 5xx)
        
        data = response.json()
        return data["choices"][0]["message"]["content"]
    
    except requests.exceptions.HTTPError as errh:
        print(f"Http Error: {errh}")
        return f"(API Ошибка Alem: {errh})"
    except Exception as e:
        print(f"Ошибка при вызове Alem API: {e}")
        return f"(API Ошибка Alem: {e})"

def generate_llm_answer(user_question, found_contexts_list):
    """
    Generates answer using RAG approach with AlemLLM.
    Formats contexts and sends to LLM for synthesis.
    """
    context_text = ""
    for i, context in enumerate(found_contexts_list):
        context_text += f"\n--- КОНТЕКСТ {i+1} ---\n"
        context_text += f"Вопрос из базы: {context.get('questions', '')}\n"
        context_text += f"Ответ из базы: {context.get('answers', '')}\n"
    
    prompt_template = f"""
    Ты — профессиональный, вежливый и дружелюбный ИИ-ассистент 
    Astana IT University (AITU).
    ... (весь ваш "умный" промпт из llm_handler.py) ...
    
    ---
    {context_text}
    ---
    
    ВОПРОС ПОЛЬЗОВАТЕЛЯ:
    {user_question}
    ---
    
    ОТВЕТ АССИСТЕНТА:
    """
    
    return _call_alem_api(prompt_template)

def generate_fallback_answer(user_question):
    """
    Fallback function when retrieval finds no relevant context.
    Sends question to LLM without RAG context, using general knowledge.
    """
    prompt_template = f"""
    Ты — полезный ИИ-ассистент Astana IT University (AITU).
    ... (весь ваш "fallback" промпт из llm_handler.py) ...

    ---
    ВОПРОС ПОЛЬЗОВАТЕЛЯ:
    {user_question}
    ---
    
    ОТВЕТ АССИСТЕНТА (С ПРЕДУПРЕЖДЕНИЕМ):
    """
    
    return _call_alem_api(prompt_template)