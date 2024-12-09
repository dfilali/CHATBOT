import logging
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
import streamlit as st

# Set up logging
# Configure the logging module to capture information about the program's execution
logging.basicConfig(level=logging.INFO)

# Load product data
# Attempt to load the product data from the 'meta.jsonl' file in JSON format
try:
    product_data = pd.read_json('meta.jsonl', lines=True)
    logging.info("Product data loaded successfully.")  # Log success
except Exception as e:
    logging.error(f"Error loading product data: {e}")  # Log error if loading fails
    raise  # Raise the exception to stop further execution if loading fails

# Process descriptions
# Convert each product description (which might be a list) into a single string and handle missing values
descriptions = product_data['description'].apply(
    lambda x: " ".join(x) if isinstance(x, list) else str(x)
).fillna("").tolist()

# Split descriptions into chunks
# Use RecursiveCharacterTextSplitter to split long descriptions into smaller, manageable chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=256,  # Maximum length of each chunk
    chunk_overlap=40  # Overlap between chunks to maintain context
)
segmented_texts = []
for description in descriptions:
    if description:
        chunks = text_splitter.split_text(description)  # Split each description into chunks
        segmented_texts.extend(chunks)  # Add chunks to the list

# Create embeddings
# Use SentenceTransformer to create vector embeddings for the chunks of text
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedding_model.encode(segmented_texts)  # Create embeddings for all text chunks
embeddings_np = np.array(embeddings)  # Convert the embeddings into a NumPy array

# Create FAISS index
# Set up a FAISS index for efficient similarity search
dimension = embeddings_np[0].shape[0]  # Get the dimension of the embeddings
index = faiss.IndexFlatL2(dimension)  # Create a flat (non-hierarchical) FAISS index
index.add(embeddings_np)  # Add the embeddings to the FAISS index

# Create documents and vector store
# Convert the text chunks into documents and store them in an in-memory docstore
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
# Create a retriever using the vector store to retrieve the most relevant documents based on a query
retriever = vector_store.as_retriever()

# Load Bloom model (or any other LLM)
# Load a pre-trained language model (Bloom in this case) for text generation
model_name = "bigscience/bloom-560m"  # Specify the model name
tokenizer = AutoTokenizer.from_pretrained(model_name)  # Load the tokenizer for the model
model = AutoModelForCausalLM.from_pretrained(model_name)  # Load the model

# Create LLM pipeline with adjustable parameters
# Create a pipeline for text generation with adjustable temperature and top-p parameters
llm_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=512,  # Max length of generated text
    truncation=True  # Enable truncation for long inputs
)

def create_improved_prompt(retrieved_docs, question, product_category):
    # Build a well-structured prompt using retrieved documents and the query
    documents = "\n".join([doc.page_content for doc in retrieved_docs])
    prompt = f"""
**Task:** Provide a comprehensive product description.

**Context:** {product_category}

**Query:** {question}

**Relevant Information:** {documents}

"""
    return prompt

def clean_response(response):
    # Clean the generated response by stripping unnecessary whitespace
    cleaned_response = response.strip()
    return cleaned_response

def generate_response(query, temperature=0.7, top_p=0.9):
    # Function to generate a response based on the user's query
    try:
        # Retrieve relevant documents based on the query
        retrieved_docs = retriever.invoke(query)
        logging.info(f"Retrieved {len(retrieved_docs)} documents")  # Log how many documents were retrieved

        if retrieved_docs:
            # Get the product category (assumed to be the first column)
            product_category = product_data['main_category'][0]
            # Generate a structured prompt with the retrieved documents and other context
            structured_prompt = create_improved_prompt(retrieved_docs, query, product_category)
            logging.info(f"Improved prompt: {structured_prompt}")  # Log the generated prompt

            # Generate a response using the language model pipeline
            response = llm_pipeline(structured_prompt, temperature=temperature, top_p=top_p)
            cleaned_response = clean_response(response[0]['generated_text'])  # Clean the generated response
            logging.info(f"Generated response: {cleaned_response}")  # Log the final response
            return cleaned_response
        else:
            logging.info("No relevant documents found.")  # Log if no documents are found
            return "No information found for your query."
    except Exception as e:
        logging.error(f"Error processing query: {e}")  # Log any error encountered
        return "An error occurred while processing your query."

# Streamlit Interface
# Streamlit is used to create a simple web interface for interacting with the model
st.title("Product Information Query")  # Display the title of the application
query = st.text_input("Enter your query:")  # Input field for the user's query

# Parameters for tuning
# Allow the user to adjust the temperature and top-p for text generation
temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
top_p = st.slider("Top-p", min_value=0.0, max_value=1.0, value=0.9, step=0.1)

if query:
    # If the user submits a query, generate a response
    response = generate_response(query, temperature=temperature, top_p=top_p)
    st.write("**Generated Response:**")
    st.write(response)  # Display the generated response in the Streamlit app
