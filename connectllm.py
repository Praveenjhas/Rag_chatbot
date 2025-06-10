import os
import fitz  
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain.load import dumps, loads
from operator import itemgetter
from langchain_huggingface import HuggingFaceEmbeddings  
from langchain_openai import ChatOpenAI 
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableMap
import re

def extract_plant_name(question, known_plants):
    """Extracts plant name from question using a list of known plants."""
    question_lower = question.lower()
    for plant in known_plants:
        if re.search(rf'\b{plant}\b', question_lower):
            return plant
    return None

def load_pdf_files(data_path):
    print("📄 Loading PDF documents...")
    documents = []
    
    for filename in os.listdir(data_path):
        if filename.lower().endswith(".pdf"):
            file_path = os.path.join(data_path, filename)
            print(f"🔍 Processing: {filename}")
            
            try:
                with fitz.open(file_path) as pdf:
                    text = ""
                    for page in pdf:
                        try:
                            page_text = page.get_text()
                            text += f"\n{page_text}\n"
                        except Exception as page_error:
                            print(f"⚠️ Error reading page: {page_error}")
                            continue
                    
                    if text.strip():
                        plant_name = os.path.splitext(filename)[0].lower()
                        documents.append(Document(
                            page_content=text,
                            metadata={
                                "source": filename, 
                                "plant": plant_name,
                            }
                        ))
                        
            except Exception as e:
                print(f"❌ Failed to process {filename}: {str(e)}")
                
    return documents

def reciprocal_rank_fusion(results: list[list], k=60):
    """Reciprocal Rank Fusion that takes multiple lists of ranked documents."""
    fused_scores = {}
    for docs in results:
        for rank, doc in enumerate(docs):
            doc_str = dumps(doc)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            fused_scores[doc_str] += 1 / (rank + k)
    reranked_results = [
        loads(doc)
        for doc, _ in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]
    return reranked_results

def filtered_retriever(question, known_plants, retriever):
    plant_name = extract_plant_name(question, known_plants)
    if plant_name:
        #print(f"🔎 Plant-specific search: {plant_name}")
        docs = retriever.invoke(question, filter={"plant": plant_name})
        if docs:
            return docs
    #print(" General guide search")
    docs = retriever.invoke(question, filter={"plant": "general"})
    return docs



embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L12-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={
        'device': 'cpu',
        'batch_size': 32,
        'normalize_embeddings': True
    }
)
db_path = "./vectorstore/db_faiss"
known_plants=['melon', 'potato', 'okra', 'lettuce', 'peanut', 'onion', 'strawberry', 'sugarcane']
if os.path.exists(db_path) and os.path.isdir(db_path) and os.listdir(db_path):
    print("✅ Database already exists. Skipping PDF processing and embedding.")
    db = Chroma(persist_directory=db_path, embedding_function=embedding_model)
else:
    documents = load_pdf_files(data_path="data")
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=300)
    chunks = splitter.split_documents(documents)
    db = Chroma.from_documents(chunks, embedding_model, persist_directory=db_path)
    db.persist()
    print(f"✅ Vector store created with {len(chunks)} chunks")

# Initialize LLM
llm = ChatOpenAI(
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key=api_key,
    model="openai/gpt-4.1",
    temperature=0.7,
    max_tokens=1024,
    streaming=False,
    max_retries=5,
    request_timeout=60
)

# Multi-query prompt
template = """You are an AI language model assistant. Your task is to generate five 
different versions of the given user question to retrieve relevant documents from a vector 
database. By generating multiple perspectives on the user question, your goal is to help
the user overcome some of the limitations of the distance-based similarity search. 
Provide these alternative questions separated by newlines. Original question: {question}"""
prompt_perspectives = ChatPromptTemplate.from_template(template)

# Decomposition Prompt
decomposition_msg = ChatPromptTemplate.from_template(
    """You are a helpful assistant that generates multiple sub-questions related to an input question. 
The goal is to break down the input into a set of sub-problems / sub-questions that can be answered in isolation. 
Generate exactly three search queries related to: {question}
Output (3 queries, each on a new line):"""
)

# Step-Back Prompt
examples = [
    {
        "input": "Could the members of The Police perform lawful arrests?",
        "output": "What can the members of The Police do?",
    },
    {
        "input": "Jan Sindel’s was born in what country?",
        "output": "What is Jan Sindel’s personal history?",
    },
]
example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{input}"),
    ("ai", "{output}"),
])
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)
step_back_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert at world knowledge. Your task is to step back and paraphrase a question to a more generic step-back question, which is easier to answer. Here are a few examples:"""),
    few_shot_prompt,
    ("user", "{question}"),
])

# Helper functions
def generate_step_back(sub_question):
    return step_back_prompt | llm | StrOutputParser()

def generate_multiquery(sub_question):
    return prompt_perspectives | llm | StrOutputParser() | (lambda x: [q.strip() for q in x.split("\n") if q.strip()])

def combine_questions(input_dict):
    question = input_dict["question"]
    # Generate three sub-questions
    sub_questions = (decomposition_msg | llm | StrOutputParser() | (lambda x: [q.strip() for q in x.split("\n") if q.strip()][:3])).invoke({"question": question})
    
    # Generate step-back questions
    step_back_questions = []
    for sub_q in sub_questions:
        step_back_q = generate_step_back(sub_q).invoke({"question": sub_q})
        step_back_questions.append(step_back_q)
    
    # Combine sub-questions and step-back questions
    combined_questions = []
    for i in range(3):
        combined_questions.append(sub_questions[i])
        combined_questions.append(step_back_questions[i])
    
    # Generate multi-query questions
    multiquery_questions = []
    for sub_q in combined_questions:
        multiquery_ques = generate_multiquery(sub_q).invoke({"question": sub_q})
        multiquery_questions.extend(multiquery_ques)
    
    return multiquery_questions

# Create the decomposition chain
final_decomposition_stepback_chain = RunnableLambda(combine_questions)

# Create the retrieval chain
def retrieval_chain_fn(input_dict):
    questions = final_decomposition_stepback_chain.invoke(input_dict)
    retriever = db.as_retriever(search_kwargs={"k": 3})
    doc_lists = [filtered_retriever(q, known_plants, retriever) for q in questions]
    return reciprocal_rank_fusion(doc_lists)

# Rerank prompt
rerank_prompt = ChatPromptTemplate.from_template("""
You are an assistant that ranks documents by their relevance to the question.
Question: {question}
Documents:
{documents}

Output:
Provide a ranked list of document indices from most relevant to least relevant, separated by commas.
""")

def rerank_documents_llm(question, documents):
    """
    Try to rerank via the LLM. On any error or bad index,
    silently fall back to the original order.
    """
    try:
        # Format the docs for the LLM
        formatted = "\n\n".join(f"[{i}]: {doc.page_content[:300]}..."
                                for i, doc in enumerate(documents))
        # Ask the LLM for a ranking
        ranked = (
            rerank_prompt
            | llm
            | StrOutputParser()
            | (lambda text: [int(idx.strip()) for idx in text.split(",")])
        ).invoke({"question": question, "documents": formatted})

        # Only keep valid indices, in order
        reranked = [documents[i] for i in ranked if 0 <= i < len(documents)]
        # If the model gave us nothing usable, fall back
        return reranked or documents

    except Exception:
        # On any error (parsing, LLM exception, index error), return original list
        return documents

# RAG prompt
rag_prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Answer the following question based on the context provided only,do you use your knowledge.

Context:
{context}

Question: {question}
""")

# Final pipeline
pipeline = (
    RunnableMap({
        "question": itemgetter("question"),
        "docs": RunnableLambda(retrieval_chain_fn)
    })
    | RunnableMap({
        "question": itemgetter("question"),
        "docs": RunnableLambda(lambda inputs: rerank_documents_llm(inputs["question"], inputs["docs"]))
    })
    | {
        "context": lambda inputs: "\n\n".join([doc.page_content for doc in inputs["docs"][:5]]),
        "question": itemgetter("question")
    }
    | rag_prompt
    | llm
    | StrOutputParser()
)

# Run the pipeline
question = "tell me about the sustainable way to plant the onion in my field"
answer = pipeline.invoke({"question": question})
print(" Final Answer:")
print(answer)
