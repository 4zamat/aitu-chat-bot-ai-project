import csv
import re
from nltk.corpus import stopwords

# Load Russian stopwords once
try:
    STOP_WORDS = set(stopwords.words('russian'))
except LookupError:
    print("NLTK stopwords не найдены. Запустите: import nltk; nltk.download('stopwords')")
    STOP_WORDS = set()

def load_data_from_csv(filepath):
    """Loads data from CSV file into a list of dictionaries."""
    data = []
    try:
        with open(filepath, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                data.append(row)
        print(f"Данные успешно загружены из {filepath}")
        return data
    except FileNotFoundError:
        print(f"ОШИБКА: Файл {filepath} не найден.")
        return None
    except Exception as e:
        print(f"Произошла непредвиденная ошибка: {e}")
        return None

def preprocess_text(text):
    """Cleans text: lowercase, remove punctuation, remove stopwords."""
    if not isinstance(text, str):
        return ""
        
    text = text.lower()
    text = re.sub(r'[^а-яa-z\s]', '', text)
    
    tokens = text.split()
    tokens = [word for word in tokens if word not in STOP_WORDS]
    
    return " ".join(tokens)