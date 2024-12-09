import numpy as np
import faiss
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.embeddings import HuggingFaceEmbeddings  # Import ajouté
from product_data_loader import load_product_data, process_descriptions

# Charger les données produit
product_data = load_product_data()
descriptions = process_descriptions(product_data)

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
    embedding_function=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),  # Utilisation de HuggingFaceEmbeddings
    index=index,
    docstore=docstore,
    index_to_docstore_id=index_to_docstore_id
)

# Create retriever
retriever = vector_store.as_retriever()

# Load Bloom model (or any other LLM)
model_name = "bigscience/bloom-560m"  # Replace with your preferred model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Create LLM pipeline with adjustable parameters
llm_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=512,
    truncation=True
)

# Function to create an improved prompt for the language model
def create_improved_prompt(retrieved_docs, question, product_category):
    documents = "\n".join([doc.page_content for doc in retrieved_docs])
    prompt = f"""
**Task:** Provide a comprehensive product description.

**Context:** {product_category}

**Query:** {question}

**Relevant Information:** {documents}

"""
    return prompt

def clean_response(response):
    cleaned_response = response.strip()
    return cleaned_response

def generate_response(query, temperature=0.7, top_p=0.9):
    try:
        retrieved_docs = retriever.invoke(query)
        if retrieved_docs:
            product_category = product_data['main_category'][0]  # Assuming category is the first column
            structured_prompt = create_improved_prompt(retrieved_docs, query, product_category)

            # Generate response with temperature and top-p
            response = llm_pipeline(structured_prompt, temperature=temperature, top_p=top_p)
            cleaned_response = clean_response(response[0]['generated_text'])
            return cleaned_response
        else:
            return "No information found for your query."
    except Exception as e:
        return "An error occurred while processing your query."
