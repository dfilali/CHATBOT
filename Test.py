import pandas as pd
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.embeddings import HuggingFaceEmbeddings

# Load embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Load product data
product_data = pd.read_json('meta.jsonl', lines=True)

# Process descriptions
descriptions = product_data['description'].apply(
    lambda x: " ".join(x) if isinstance(x, list) else str(x)
).fillna("").tolist()

# Split descriptions into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=256,
    chunk_overlap=40
)
segmented_texts = []
for description in descriptions:
    if description:
        chunks = text_splitter.split_text(description)
        segmented_texts.extend(chunks)

# Create embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedding_model.encode(segmented_texts)
embeddings_np = np.array(embeddings)

# Create FAISS index
dimension = embeddings_np[0].shape[0]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings_np)

# Create documents and vector store
documents = [Document(page_content=text) for text in segmented_texts]
docstore = InMemoryDocstore({str(i): doc for i, doc in enumerate(documents)})
index_to_docstore_id = {i: str(i) for i in range(len(documents))}
vector_store = FAISS(
    embedding_function=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
    index=index,
    docstore=docstore,
    index_to_docstore_id=index_to_docstore_id
)

# Create retriever
retriever = vector_store.as_retriever()

# Load Bloom model
model_name = "bigscience/bloom-560m"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Create Bloom pipeline
bloom_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=512,
    truncation=True
)

# Improved prompt engineering function
def create_improved_prompt(retrieved_docs, question, product_category):
    documents = "\n".join([doc.page_content for doc in retrieved_docs])
    prompt = f"""
    **Product Category:** {product_category}

    **Prompt:**
    {question}

    **Relevant Information:**
    {documents}

    **Response:**
    Provide a comprehensive and informative answer based on the given information. If the answer cannot be found in the information, state "I don't know."
    """
    return prompt

# Example query
query = "who is macron ?"

# Retrieve relevant documents
try:
    retrieved_docs = retriever.get_relevant_documents(query)
    print(f"Retrieved documents ({len(retrieved_docs)}):")
    for doc in retrieved_docs[:3]:
        print(doc.page_content)
except Exception as e:
    print(f"Error retrieving documents: {e}")
    retrieved_docs = []

# Generate response
if retrieved_docs:
    product_category = product_data['main_category'][0]  # Assuming category is the first column
    structured_prompt = create_improved_prompt(retrieved_docs, query, product_category)
    print(f"Improved prompt: {structured_prompt}")

    try:
        response = bloom_pipeline(structured_prompt)
        print(f"Generated response: {response[0]['generated_text']}")
    except Exception as e:
        print(f"Error generating text with Bloom: {e}")
else:
    print("No documents found to process the query.")