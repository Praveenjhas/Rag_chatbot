# 🌾 Agricult-AIcult RAG Bot

A Retrieval-Augmented Generation (RAG) bot designed to assist farmers and agriculture experts by answering crop-related queries using a document-based question-answering system. This project utilizes advanced LLM techniques, document retrieval, multi-query decomposition, reranking, and automatic evaluation using RAGAS.

## 📌 Features

- 💬 Accepts user questions about crops (e.g., onion cultivation)
- 📄 Extracts context from domain-specific PDFs
- 🔍 Performs:
  - Sub-question decomposition
  - Step-back question generation
  - Multi-query expansion
  - **Reciprocal Rank Fusion (RRF)**-based retrieval
  - LLM-based reranking
- 🧠 Generates answers with GPT-3.5 via OpenRouter
- ✅ Automatically evaluates responses using **RAGAS** (Faithfulness, Context Precision, Context Recall)
- 📊 Exports evaluation results to **Excel**

---

## 🧰 Tech Stack

| Component       | Tool/Library                                |
|----------------|----------------------------------------------|
| Embedding       | `sentence-transformers/all-MiniLM-L12-v2`   |
| Vector Store    | `Chroma` + FAISS                            |
| LLM             | `GPT-3.5-turbo` via [OpenRouter](https://openrouter.ai) |
| Evaluation      | `RAGAS` (Faithfulness, Context Precision, Context Recall) |
| PDF Parsing     | `PyMuPDF (fitz)`                            |
| Query Handling  | `LangChain` + custom chains                 |
| Excel Export    | `pandas` + `openpyxl`                       |

---

## 📂 Directory Structure
Agricult-AIcult-RAG
├── data/ # Folder containing crop PDFs
├── vectorstore/ # Chroma DB vector store (auto-created)
├── ragas_evaluation_report.xlsx # Evaluation results (auto-generated)
├── main.py # Full RAG pipeline script
├── requirements.txt # Required Python dependencies
└── README.md # You're here!
2. Install dependencies
Make sure you have Python 3.11.9+ installed.
pip install -r requirements.txt
If you face issues with datasets, install it manually:
pip install datasets
3. Add your PDF files
Place your crop-specific PDFs (e.g., onion.pdf, general.pdf) inside the data/ folder. Each file should be named after the crop it covers (lowercase, no spaces).

4. Add your OpenRouter & OpenAI Keys
In main.py, replace:
openai_api_key="sk-..."  # for OpenRouter
os.environ["OPENAI_API_KEY"] = "sk-..."  # for RAGAS

🙏 Acknowledgments
LangChain
OpenRouter
RAGAS
HuggingFace Sentence Transformers
