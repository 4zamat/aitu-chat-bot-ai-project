# AITU FAQ Chatbot

This project is a FAQ chatbot for Astana IT University (AITU) built as a Python final project. It uses a Retrieval-Augmented Generation (RAG) pipeline to answer user questions based on a local `.csv` file.

The core logic is built from scratch, demonstrating a custom RAG implementation.

### How it works:

1.  **Retrieve:** `sentence-transformers` (SBERT) performs a semantic search on the university's FAQ data to find the 3 most relevant context snippets (`k=3`).
2.  **Generate:** The Google Gemini API receives this context and the user's question, then generates a synthesized, natural-language answer.
3.  **Disambiguation (Plan C):** If the initial search fails to find a high-confidence match (e.g., for a vague, single-word query like "dormitory"), the bot asks for clarification. It uses Streamlit's `session_state` to "remember" the topic, then merges the user's next message (e.g., "cost") for a successful, targeted search (`cost dormitory`).
4.  **Interface:** `streamlit` provides the chat UI and manages the chat history and memory.



## 🚀 Running Locally

### 1. Clone the Repository
```bash
git clone [https://github.com/4zamat/aitu-chat-bot-ai-project.git](https://github.com/4zamat/aitu-chat-bot-ai-project.git)
cd aitu-chat-bot-ai-project
````

### 2\. Install Dependencies

It is highly recommended to use a virtual environment.

```bash
# Create a virtual environment
python -m venv venv
# Activate it (macOS/Linux)
source venv/bin/activate
# (Windows)
# venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 3\. Set Up Your API Key

This project requires a Google AI Studio API key.

1.  Create a new file in the root directory named **`.env`**
2.  Add your API key to this file:
    ```
    GOOGLE_API_KEY="your-api-key-here"
    ```

The `.gitignore` file is configured to secure this key.

### 4\. Add Your Data

1.  Place your FAQ data file (e.g., `aitu_faq.csv`) in the root directory.
2.  Update the `DATA_FILE` variable in `app.py` to match your file's name.

### 5\. Run the App

```bash
streamlit run app.py
```

Streamlit will automatically open the application in your browser.

-----

## 🛠️ Core Technologies Used

  * **Python 3.10+**
  * **Streamlit:** For the web interface and session state management.
  * **Sentence-Transformers (SBERT):** For semantic search/retrieval.
  * **Google Generative AI:** For the LLM (Gemini) generation step.
  * **Pandas:** For loading the initial `.csv` data.
  * **python-dotenv:** For secure API key management.

## Project Structure

```
.
├── app.py              # The main Streamlit application
├── chatbot_logic_v3.py  # SBERT retrieval logic (k=3, thresholding)
├── llm_handler.py      # Gemini API logic (RAG & Fallback prompters)
├── data_handler.py     # CSV loading function
├── aitu_faq.csv        # (Your FAQ data file)
├── requirements.txt    # Project dependencies
├── .env                # (Must be created for API key)
└── .gitignore
```