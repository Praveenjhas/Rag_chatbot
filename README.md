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

## 📂 Directory Structure

Agricult-AIcult-RAG
├── data/ # Folder containing crop PDFs
├── vectorstore/ # Chroma DB vector store (auto-created)
├── ragas_evaluation_report.xlsx # Evaluation results (auto-generated)
├── main.py # Full RAG pipeline script
├── requirements.txt # Required Python dependencies
└── README.md # You're here!


---

## 🔧 Setup Instructions

### 1️⃣ Install Dependencies

Make sure you have **Python 3.11.9+** installed.

Install all required packages:

```bash
pip install -r requirements.txt
* installed.

If you face issues with datasets, install it separately:
pip install datasets
2️⃣ Add Your PDF Files
Place your crop-specific PDF files inside the data/ directory.
Each file should be named according to the crop it covers, all lowercase with no spaces. For example:
data/
├── onion.pdf
├── wheat.pdf
├── tomato.pdf
3️⃣ Add Your OpenRouter & OpenAI Keys
In main.py, replace the following lines with your actual API keys:
# OpenRouter key for LLM calls
openai_api_key = "sk-..."  

# OpenAI key for RAGAS evaluation (used internally by ragas)
import os
os.environ["OPENAI_API_KEY"] = "sk-..."
🙏 Acknowledgments
LangChain

OpenRouter

RAGAS

HuggingFace Sentence Transformers
