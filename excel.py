import os
import fitz  # PyMuPDF
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
from ragas import evaluate
from ragas.metrics import faithfulness, context_precision, context_recall
from datasets import Dataset
import pandas as pd
from datetime import datetime


def extract_plant_name(question, known_plants):
    question_lower = question.lower()
    for plant in known_plants:
        if re.search(rf'\b{plant}\b', question_lower):
            return plant
    return None

def load_pdf_files(data_path):
    documents = []
    for filename in os.listdir(data_path):
        if filename.lower().endswith(".pdf"):
            file_path = os.path.join(data_path, filename)
            print(f"Processing: {filename}")
            try:
                with fitz.open(file_path) as pdf:
                    text = ""
                    for page in pdf:
                        page_text = page.get_text()
                        text += f"\n{page_text}\n"
                    if text.strip():
                        plant_name = os.path.splitext(filename)[0].lower()
                        documents.append(Document(
                            page_content=text,
                            metadata={"source": filename, "plant": plant_name}
                        ))
            except Exception as e:
                print(f"Failed to process {filename}: {str(e)}")
    return documents

def reciprocal_rank_fusion(results: list[list], k=60):
    fused_scores = {}
    for docs in results:
        for rank, doc in enumerate(docs):
            doc_str = dumps(doc)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            fused_scores[doc_str] += 1 / (rank + k)
    reranked_results = [loads(doc) for doc, _ in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)]
    return reranked_results

def filtered_retriever(question, known_plants, retriever):
    plant_name = extract_plant_name(question, known_plants)
    if plant_name:
        docs = retriever.invoke(question, filter={"plant": plant_name})
        if docs:
            return docs
    docs = retriever.invoke(question, filter={"plant": "general"})
    return docs

# Setup embedding and vector store
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L12-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'device': 'cpu', 'batch_size': 32, 'normalize_embeddings': True}
)
documents = load_pdf_files(data_path="data")
known_plants = list({doc.metadata["plant"] for doc in documents})
db_path = "./vectorstore/db_faiss"
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.split_documents(documents)
db = Chroma.from_documents(chunks, embedding_model, persist_directory=db_path)
print(f"Vector store created with {len(chunks)} chunks")
retriever = db.as_retriever(search_kwargs={"k": 3})

# Initialize LLM
llm = ChatOpenAI(
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key=api_key,
    model="openai/gpt-4.1",
    temperature=0.5,
    max_tokens=1024,
    streaming=False,
    max_retries=3,
    request_timeout=60
)

template = """You are an AI language model assistant. Your task is to generate five 
different versions of the given user question to retrieve relevant documents from a vector 
database. By generating multiple perspectives on the user question, your goal is to help
the user overcome some of the limitations of the distance-based similarity search. 
Provide these alternative questions separated by newlines. Original question: {question}"""
prompt_perspectives = ChatPromptTemplate.from_template(template)

decomposition_msg = ChatPromptTemplate.from_template(
    """You are a helpful assistant that generates multiple sub-questions related to an input question. 
The goal is to break down the input into a set of sub-problems / sub-questions that can be answered in isolation. 
Generate exactly three search queries related to: {question}
Output (3 queries, each on a new line):"""
)

examples = [
    {"input": "Could the members of The Police perform lawful arrests?", "output": "What can the members of The Police do?"},
    {"input": "Jan Sindel's was born in what country?", "output": "What is Jan Sindel's personal history?"}
]
example_prompt = ChatPromptTemplate.from_messages([("human", "{input}"), ("ai", "{output}")])
few_shot_prompt = FewShotChatMessagePromptTemplate(example_prompt=example_prompt, examples=examples)
step_back_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert at world knowledge. Your task is to step back and paraphrase a question to a more generic step-back question, which is easier to answer. Here are a few examples:"""),
    few_shot_prompt,
    ("user", "{question}")
])

def generate_step_back(sub_question):
    return step_back_prompt | llm | StrOutputParser()

def generate_multiquery(sub_question):
    return prompt_perspectives | llm | StrOutputParser() | (lambda x: [q.strip() for q in x.split("\n") if q.strip()])

def combine_questions(input_dict):
    question = input_dict["question"]
    sub_questions = (decomposition_msg | llm | StrOutputParser() | (lambda x: [q.strip() for q in x.split("\n") if q.strip()][:3])).invoke({"question": question})
    step_back_questions = [generate_step_back(sub_q).invoke({"question": sub_q}) for sub_q in sub_questions]
    combined_questions = []
    for i in range(len(sub_questions)):
        combined_questions.append(sub_questions[i])
        combined_questions.append(step_back_questions[i])
    multiquery_questions = []
    for sub_q in combined_questions:
        multiquery_ques = generate_multiquery(sub_q).invoke({"question": sub_q})
        multiquery_questions.extend(multiquery_ques)
    return multiquery_questions

final_decomposition_stepback_chain = RunnableLambda(combine_questions)

def retrieval_chain_fn(input_dict):
    questions = final_decomposition_stepback_chain.invoke(input_dict)
    retriever = db.as_retriever(search_kwargs={"k": 3})
    doc_lists = [filtered_retriever(q, known_plants, retriever) for q in questions]
    return reciprocal_rank_fusion(doc_lists)

rerank_prompt = ChatPromptTemplate.from_template("""
You are an assistant that ranks documents by their relevance to the question.
Question: {question}
Documents:
{documents}

Output:
Provide a ranked list of document indices from most relevant to least relevant, separated by commas.
""")

def rerank_documents_llm(question, documents):
    try:
        formatted = "\n\n".join(f"[{i}]: {doc.page_content[:300]}..." for i, doc in enumerate(documents))
        ranked = (
            rerank_prompt
            | llm
            | StrOutputParser()
            | (lambda text: [int(idx.strip()) for idx in text.split(",")])
        ).invoke({"question": question, "documents": formatted})
        reranked = [documents[i] for i in ranked if 0 <= i < len(documents)]
        return reranked or documents
    except Exception:
        return documents

rag_prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Answer the following question based on the context provided only,do you use your knowledge.

Context:
{context}

Question: {question}
""")

# Define the pipeline
pipeline = (
    RunnableMap({"question": itemgetter("question"), "docs": RunnableLambda(retrieval_chain_fn)})
    | RunnableMap({"question": itemgetter("question"), "docs": RunnableLambda(lambda inputs: rerank_documents_llm(inputs["question"], inputs["docs"]))})
    | {"context": lambda inputs: "\n\n".join([doc.page_content for doc in inputs["docs"][:5]]), "question": itemgetter("question")}
    | rag_prompt
    | llm
    | StrOutputParser()
)

# Define test cases
test_cases = [
        {
            "question": "What are the optimal soil conditions for growing onions?",
            "ground_truth": (
                "The best soil for successful onion cultivation is deep, friable loam and alluvial soils "
                "with good drainage, moisture-holding capacity and sufficient organic matter. "
                "The optimum pH range, regardless of soil type, is 6.0 - 7.5, but onion can also be grown "
                "in mild alkaline soils."
            )
        },
        {
            "question": "What fertilizers should I use for onion cultivation?",
            "ground_truth": (
                "For a successful crop, follow these recommendations: "
                "1) Add farmyard manure (10 tons per acre) and incorporate well during the last ploughing. "
                "2) Add 45 kg per acre of urea, 135 kg per acre of SSP and 35 kg per acre of MOP when ridges are opened. "
                "3) An additional dose of urea should be applied later, one month after transplantation. "
                "4) Onions are also very sensitive to zinc deficiencies. Apply 50 kg per acre of zinc sulfate during the last ploughing."
            )
        },
        {
            "question": "How do I prepare seedbeds for onion nursery?",
            "ground_truth": (
                "Here are the 6 steps: "
                "Step 1 - Choose an open, protected, sunny and well-drained area of the field. "
                "Step 2 - Mark the seedbed plots (2-3 m x 1 m) and mix the soil thoroughly with a rake. "
                "Step 3 - Incorporate well-decomposed farmyard manure at a rate of 4-5 kg/m2 into soils. "
                "Step 4 - Form raised seedbeds 15 cm or higher, 2-3 m in length and 80-100 cm wide. "
                "Step 5 - Cover the soil with a plastic sheet and leave it for 10 days (solarization). "
                "Step 6 - Set up a net-tunnels structure above the seedbeds with 32-to 60-mesh nylon netting."
            )
        },
        {
            "question": "What climate conditions are best for onion growing?",
            "ground_truth": (
                "Onion is a temperate crop but can be grown under a wide range of climatic conditions. "
                "The best performance can be obtained in mild weather without the extremes of cold and heat "
                "and excessive rainfall. It requires about 70% relative humidity for good growth. "
                "It can grow well in places where the average annual rainfall is 650-750 mm. "
                "Onion crops need lower temperature and shorter daylight during vegetative growth "
                "while during bulb development it needs higher temperature and longer daylight."
            )
        },
        {
            "question": "How do I prevent damping off in onion seedlings?",
            "ground_truth": (
                "To prevent damping-off your plants, make sure to: "
                "1) Improve drainage of the soil before planting. "
                "2) Not plant seedlings too deep when transplanting. "
                "3) Remove infected plants as the first symptoms appear. "
                "4) Always water in the morning so that the soil is dry by evening. "
                "5) Not inadvertently transport mud from one field to another."
            )
        },
        {
            "question": "When should I transplant onion seedlings?",
            "ground_truth": (
                "Seedlings are ready for transplanting around 35-40 days after sowing during the Kharif season "
                "and 45-50 DAS during late Kharif and Rabi seasons. At the time of transplanting, trim the top "
                "off the seedling to ensure stronger plants. Plant seedlings at 10 cm between plants and 15 cm between the rows."
            )
        },
        {
            "question": "How do I cure onions after harvesting?",
            "ground_truth": (
                "Under mild weather conditions, bulbs can be cured directly in the field after being pulled out. "
                "Under hot weather conditions, bulbs should be removed from the field and left to cure in the shade. "
                "During the kharif season, bulbs are cured for 2-3 weeks keeping the tops of the plants. "
                "During the rabi season, bulbs are cured in the field for 3-5 days, then tops are cut off "
                "leaving 2.0-2.5 cm above the bulb and again cured for 7-10 days away from the field."
            )
        },
        {
            "question": "What are the symptoms of purple blotch in onions?",
            "ground_truth": (
                "Small, irregular, sunken and whitish specks appear on older leaves and flower stalks. "
                "At high relative humidity, these lesions develop into elliptical brown or purple blotches, "
                "with concentric light and dark zones on their centre. Leaves and flower stalks wilt and die."
            )
        },
        {
            "question": "How do I know when onions are ready for harvest?",
            "ground_truth": (
                "When the bulbs that develop from the leaf bases of onions are fully formed, "
                "the leafy green tops begin to yellow and eventually collapse at a point just above "
                "the top of the bulb, leaving an upright short neck. When the tops 'go down' in this way, "
                "it indicates that the bulbs are ready for harvesting. Harvest by pulling out plants "
                "when tops are drooping but still green."
            )
        }
    ]

#  RAGAS Evaluation for multiple test cases
def run_ragas_evaluation():
    print(" Starting RAGAS evaluation for multiple test cases...")
    
    
    # Prepare lists for batch evaluation
    questions = []
    answers = []
    contexts_list = []
    ground_truths = []
    
    print(f" Processing {len(test_cases)} test cases...")
    
    for i, test_case in enumerate(test_cases):
        print(f"Processing test case {i+1}/{len(test_cases)}: {test_case['question'][:50]}...")
        
        try:
            result = pipeline.invoke({"question": test_case["question"]})
            # Retrieve contexts
            retrieved_docs = retrieval_chain_fn({"question": test_case["question"]})
            contexts = [doc.page_content for doc in retrieved_docs[:5]]
            # Add to lists
            questions.append(test_case["question"])
            answers.append(result)
            contexts_list.append(contexts)
            ground_truths.append(test_case["ground_truth"])
            
        except Exception as e:
            print(f" Error processing test case {i+1}: {str(e)}")
            # Add empty/default values to maintain list consistency
            questions.append(test_case["question"])
            answers.append("Error generating answer")
            contexts_list.append(["No context retrieved"])
            ground_truths.append(test_case["ground_truth"])
    
    # Prepare data for RAGAS
    evaluation_data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts_list,
        "ground_truth": ground_truths
    }
    dataset = Dataset.from_dict(evaluation_data)
    metrics = [faithfulness, context_precision, context_recall]
    
    print("Running RAGAS evaluation...")
    # Run evaluation
    evaluation_results = evaluate(dataset, metrics=metrics, llm=llm)
    
    return evaluation_results, questions, answers, ground_truths

def create_excel_report(evaluation_results, questions, answers, ground_truths):
    print("Excel report...")
    
    # Convert EvaluationResult to DataFrame and extract scores
    results_df = evaluation_results.to_pandas()
    
    #  individual scores for each test case
    faithfulness_scores = results_df['faithfulness'].tolist()
    context_precision_scores = results_df['context_precision'].tolist()
    context_recall_scores = results_df['context_recall'].tolist()
    
    # detailed results DataFrame
    detailed_results = []
    for i in range(len(questions)):
        detailed_results.append({
            'Test_Case_ID': f'TC_{i+1:02d}',
            'Question': questions[i],
            'Generated_Answer': answers[i],
            'Ground_Truth': ground_truths[i],
            'Faithfulness': faithfulness_scores[i] if i < len(faithfulness_scores) else 0.0,
            'Context_Precision': context_precision_scores[i] if i < len(context_precision_scores) else 0.0,
            'Context_Recall': context_recall_scores[i] if i < len(context_recall_scores) else 0.0,
            'Average_Score': (
                (faithfulness_scores[i] if i < len(faithfulness_scores) else 0.0) + 
                (context_precision_scores[i] if i < len(context_precision_scores) else 0.0) + 
                (context_recall_scores[i] if i < len(context_recall_scores) else 0.0)
            ) / 3
        })
    
    detailed_df = pd.DataFrame(detailed_results)
    
    # Create summary statistics
    summary_stats = {
        'Metric': ['Faithfulness', 'Context_Precision', 'Context_Recall', 'Overall_Average'],
        'Average_Score': [
            sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else 0.0,
            sum(context_precision_scores) / len(context_precision_scores) if context_precision_scores else 0.0,
            sum(context_recall_scores) / len(context_recall_scores) if context_recall_scores else 0.0,
            detailed_df['Average_Score'].mean()
        ],
        'Min_Score': [
            min(faithfulness_scores) if faithfulness_scores else 0.0,
            min(context_precision_scores) if context_precision_scores else 0.0,
            min(context_recall_scores) if context_recall_scores else 0.0,
            detailed_df['Average_Score'].min()
        ],
        'Max_Score': [
            max(faithfulness_scores) if faithfulness_scores else 0.0,
            max(context_precision_scores) if context_precision_scores else 0.0,
            max(context_recall_scores) if context_recall_scores else 0.0,
            detailed_df['Average_Score'].max()
        ],
        'Std_Deviation': [
            pd.Series(faithfulness_scores).std() if faithfulness_scores else 0.0,
            pd.Series(context_precision_scores).std() if context_precision_scores else 0.0,
            pd.Series(context_recall_scores).std() if context_recall_scores else 0.0,
            detailed_df['Average_Score'].std()
        ]
    }
    
    summary_df = pd.DataFrame(summary_stats)
    
    # Create evaluation metadata
    metadata = {
        'Information': [
            'Evaluation Date',
            'Total Test Cases',
            'Model Used',
            'Embedding Model',
            'Metrics Evaluated',
            'Vector Store Chunks'
        ],
        'Value': [
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            len(test_cases),
            'gpt-3.5-turbo (via OpenRouter)',
            'sentence-transformers/all-MiniLM-L12-v2',
            'Faithfulness, Context Precision, Context Recall',
            len(chunks)
        ]
    }
    
    metadata_df = pd.DataFrame(metadata)
    
    # Save to Excel with multiple sheets
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'RAG_Evaluation_Report_{timestamp}.xlsx'
    
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        # Write detailed results
        detailed_df.to_excel(writer, sheet_name='Detailed_Results', index=False)
        
        # Write summary statistics
        summary_df.to_excel(writer, sheet_name='Summary_Statistics', index=False)
        
        # Write metadata
        metadata_df.to_excel(writer, sheet_name='Evaluation_Metadata', index=False)
        
        # Format the Excel sheets
        for sheet_name in writer.sheets:
            worksheet = writer.sheets[sheet_name]
            for column in worksheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                adjusted_width = min(max_length + 2, 50)  # Cap at 50 characters
                worksheet.column_dimensions[column_letter].width = adjusted_width
    
    print(f"Excel report saved as: {filename}")
    return filename, detailed_df, summary_df

if __name__ == "__main__":
    try:
        # Run the evaluation
        print(" Starting RAG System Evaluation...")
        results, questions, answers, ground_truths = run_ragas_evaluation()
        
        print("\nRAGAS Evaluation Results:")
        print("="*50)
        print(f"Evaluation completed successfully!")
        
        # Convert to pandas DataFrame for easier access
        results_df = results.to_pandas()
        print(f"\nResults DataFrame:")
        print(results_df)
        
        # Print average scores
        print(f"\nAverage Scores:")
        for column in results_df.columns:
            if column in ['faithfulness', 'context_precision', 'context_recall']:
                avg_score = results_df[column].mean()
                print(f"{column}: {avg_score:.4f}")
        
        # Create Excel report
        excel_file, detailed_df, summary_df = create_excel_report(results, questions, answers, ground_truths)
        
        print(f"\nSummary Statistics:")
        print("="*50)
        print(summary_df.to_string(index=False))
        
        print(f"\n Evaluation completed successfully!")
        print(f"Detailed report saved to: {excel_file}")
        
    except Exception as e:
        print(f" Error during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()