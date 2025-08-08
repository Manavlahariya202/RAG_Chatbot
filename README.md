# Production-Ready RAG Agent Chatbot

This project is a production-worthy, agent-based AI Chatbot that answers user questions based on the content of one or more uploaded PDF documents. It features a sophisticated, memory-enabled agent that can accurately retrieve information and cite its sources.



---

## Key Features

* **Retrieval-Augmented Generation (RAG)**: Leverages an advanced agent to find the most relevant information from your documents before answering a question.
* **Multi-Document Support**: Upload and query multiple PDF files simultaneously.
* **Metadata Filtering**: Dynamically filter the search to specific documents for more precise answers.
* **Conversational Memory**: The chatbot remembers the context of the conversation for a natural, multi-turn dialogue.
* **Source Citation**: The agent is instructed to cite the source document and page number for each finding, ensuring verifiability.
* **Modern Architecture**: Built with the latest LangChain standards, including `RunnableWithMessageHistory` for efficient and robust memory management.

---

## Tech Stack

* **Backend**: Python
* **LLM Framework**: LangChain
* **LLM Provider**: OpenAI (GPT-4o)
* **Vector Database**: ChromaDB
* **PDF Parsing**: PyMuPDF
* **Frontend**: Streamlit

---

##  Setup & Installation

Follow these steps to set up and run the project locally.

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd <your-repo-directory>
```

### 2. Create a Virtual Environment
It's highly recommended to use a virtual environment to manage dependencies.

* **On Windows:**
    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    ```

### 3. Install Dependencies
Install all the required Python packages from the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables
Create a file named `.env` in the root directory of the project and add your OpenAI API key:

```
# .env
OPENAI_API_KEY="..."
```

---

## Usage

1.  **Run the Streamlit App:**
    Start the application from your terminal:
    ```bash
    streamlit run app.py
    ```

2.  **Interact with the Chatbot:**
    * Open the URL provided by Streamlit in your browser (usually `http://localhost:8501`).
    * Use the sidebar to upload one or more PDF documents.
    * Click the **"Process Documents"** button and wait for it to finish.
    * Type your questions into the chat box at the bottom and get answers directly from your documents.

## UI
<img width="1914" height="968" alt="image" src="https://github.com/user-attachments/assets/e40d77ff-26d6-49de-8cc5-e5f4baae2331" />
