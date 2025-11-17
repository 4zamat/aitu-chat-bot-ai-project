import pandas as pd
import google.generativeai as genai
import time
from tqdm import tqdm 
import re
from dotenv import load_dotenv
import os

load_dotenv()

# Configure API key (same as llm_handler)
try:
    YOUR_API_KEY = os.getenv("GOOGLE_API_KEY")

    if YOUR_API_KEY is None:
        print("Ошибка: GOOGLE_API_KEY не найден. Создайте .env файл.")
    genai.configure(api_key=YOUR_API_KEY)
except Exception as e:
    YOUR_API_KEY = None

model = genai.GenerativeModel(model_name="gemini-2.0-flash")

def create_golden_list_prompt(dirty_tag_list):
    """
    Creates prompt for LLM to analyze all unique dirty tags and generate a "Golden List".
    The Golden List contains 10-15 high-level categories that cover all tags.
    """
    tags_as_string = "\n".join(dirty_tag_list)
    
    return f"""
    Ты — ИИ-ассистент по анализу данных.
    
    Проанализируй следующий список "грязных" тегов из базы данных.
    Твоя задача — выявить 10-15 **основных, высокоуровневых тем** (на русском языке), которые охватывают этот список.
    
    ПРАВИЛА:
    1. Игнорируй мусор, кавычки и странные форматы.
    2. Объединяй похожие темы (например, "Жилье" и "Общежитие").
    3. Ответ должен быть **только** списком слов, разделенных запятой.
    4. Не пиши ничего, кроме этого списка.
    
    Пример ответа:
    Общее, Поступление, Экзамены, Общежитие, Кампус, Учебный процесс, Оплата, Прочее
    
    ---
    СПИСОК "ГРЯЗНЫХ" ТЕГОВ:
    {tags_as_string}
    ---
    
    "ЗОЛОТОЙ СПИСОК" (10-15 тем):
    """

def create_cleaning_prompt(dirty_tag, golden_list_string):
    """
    Creates prompt to clean a single tag by mapping it to categories from the Golden List.
    """
    return f"""
    Ты — ИИ-ассистент по очистке данных. Твоя задача — стандартизировать "грязный" тег.

    ПРАВИЛА:
    1. Проанализируй "ГРЯЗНЫЙ ТЕГ".
    2. Выбери из "ЗОЛОТОГО СПИСКА" один или несколько (через запятую) 
       наиболее подходящих тегов.
    3. Если ничего не подходит, верни "Прочее".
    4. Не придумывай свои теги.
    5. Ответ должен быть **только** списком тегов (например: "Общежитие, Жилье").

    ---
    ЗОЛОТОЙ СПИСОК:
    {golden_list_string}
    ---
    
    ГРЯЗНЫЙ ТЕГ:
    "{dirty_tag}"
    ---

    ЧИСТЫЕ ТЕГИ:
    """

def generate_golden_list(dirty_tag_list):
    """
    Step 1: Generates Golden List by analyzing all unique dirty tags.
    Returns cleaned string of categories separated by commas.
    """
    prompt = create_golden_list_prompt(dirty_tag_list)
    try:
        response = model.generate_content(prompt)
        # Clean LLM response: keep only words, commas, and spaces
        clean_list_string = re.sub(r'[^\w\s,]', '', response.text).strip()
        print(f"--- LLM СГЕНЕРИРОВАЛ 'ЗОЛОТОЙ СПИСОК' ---\n{clean_list_string}\n----------------------------------")
        return clean_list_string
    except Exception as e:
        print(f'Критическая ошибка при генерации "Золотого Списка": {e}')
        return None

def clean_tag_with_llm(dirty_tag, golden_list_string):
    """
    Step 2 (called in loop): Cleans a single tag by mapping it to Golden List categories.
    """
    if not dirty_tag or pd.isna(dirty_tag):
        return "Прочее"
    
    prompt = create_cleaning_prompt(dirty_tag, golden_list_string)
    try:
        response = model.generate_content(prompt)
        clean_tag = response.text.strip().strip('"')
        return clean_tag
    except Exception as e:
        print(f"Ошибка API при обработке тега '{dirty_tag}': {e}")
        return "Прочее" 

def main_cleaner():
    """
    Main data cleaning pipeline:
    1. Extract unique dirty tags
    2. Generate Golden List (high-level categories)
    3. Clean each tag using Golden List
    4. Apply mapping to entire dataset
    5. Save cleaned data
    """
    if YOUR_API_KEY is None:
        print("Ошибка: API-ключ не настроен.")
        return

    INPUT_FILE = "data/QA_addmissionAitu.csv" 
    OUTPUT_FILE = "data/QA_cleaned.csv"
    COLUMN_TO_CLEAN = "classes" 

    print(f"Загружаю данные из {INPUT_FILE}...")
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"Ошибка: Файл {INPUT_FILE} не найден.")
        return

    # Extract unique dirty tags
    unique_dirty_tags = df[COLUMN_TO_CLEAN].dropna().unique()
    print(f"Найдено {len(unique_dirty_tags)} уникальных 'грязных' тегов.")
    
    # Generate Golden List (high-level categories)
    golden_list_string = generate_golden_list(unique_dirty_tags)
    if golden_list_string is None:
        print("Не удалось сгенерировать 'Золотой Список'. Выход.")
        return
        
    # Clean each unique tag using Golden List
    mapping = {}
    print("Начинаю очистку каждого тега с помощью LLM...")
    for tag in tqdm(unique_dirty_tags):
        clean_version = clean_tag_with_llm(tag, golden_list_string)
        mapping[tag] = clean_version
        time.sleep(1.1)  # Rate limiting for API

    # Apply mapping to entire dataset
    print("Очистка завершена. Применяю маппинг к файлу...")
    df[COLUMN_TO_CLEAN] = df[COLUMN_TO_CLEAN].map(mapping).fillna("Прочее")

    # Save cleaned data
    df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
    print(f"Готово! Очищенные данные сохранены в {OUTPUT_FILE}")

if __name__ == "__main__":
    main_cleaner()