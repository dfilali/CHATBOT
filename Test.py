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

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Charger les données depuis le fichier JSONL
product_data = pd.read_json('meta.jsonl', lines=True)
print(product_data.head())

# Extraire les descriptions de chaque produit
descriptions = product_data['description'].fillna("").tolist()
print(descriptions)

# Configurer le séparateur
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=256,  # Taille de chaque chunk
    chunk_overlap=40  # Chevauchement entre les chunks
)

# Diviser les descriptions en chunks
segmented_texts = []
for description in descriptions:
    if description:  # Si la description existe
        chunks = text_splitter.split_text(" ".join(description))
        segmented_texts.extend(chunks)

print(f"Nombre total de chunks créés : {len(segmented_texts)}")
print("Exemple de chunk :", segmented_texts[:2])

# Utiliser un modèle d'embeddings de Sentence-Transformer
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Générer les embeddings pour les segments
embeddings = embedding_model.encode(segmented_texts)

# Convertir les embeddings en numpy.ndarray
embeddings_np = np.array(embeddings)

# Vérifier la dimension des embeddings
dimension = embeddings_np[0].shape[0]  # Dimension des vecteurs d'embeddings
print(f"Dimension des embeddings : {dimension}")

# Créer un index FAISS pour les embeddings
index = faiss.IndexFlatL2(dimension)  # Index basé sur la distance euclidienne

# Ajouter les embeddings à l'index
index.add(embeddings_np)

print(f"Nombre de vecteurs indexés : {index.ntotal}")

# Créer les documents
documents = [Document(page_content=text) for text in segmented_texts]

# Créer un mapping d'index pour les documents
docstore = InMemoryDocstore({str(i): doc for i, doc in enumerate(documents)})
index_to_docstore_id = {i: str(i) for i in range(len(documents))}

# Créer un vector store avec FAISS
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=docstore,
    index_to_docstore_id=index_to_docstore_id
)

# Configurer un retriever pour rechercher les descriptions pertinentes
retriever = vector_store.as_retriever()

# Charger le modèle Bloom depuis Hugging Face
model_name = "bigscience/bloom-560m"  # Vous pouvez utiliser un modèle plus grand, si besoin
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Configurer le pipeline de génération
bloom_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, max_length=512)

# Fonction pour formuler le prompt structuré
def create_structured_prompt(retrieved_docs, question):
    documents = "\n".join([doc.page_content for doc in retrieved_docs])
    prompt = f"""
    Vous êtes un assistant intelligent conçu pour répondre aux questions des utilisateurs en utilisant exclusivement les informations contenues dans les documents fournis ci-dessous.
    Vous ne devez **pas** faire appel à vos connaissances internes ni générer d'informations supplémentaires ou incorrectes qui ne sont pas présentes dans ces documents.

    Voici les instructions à suivre :
    1. Utilisez **uniquement les informations présentes dans les documents fournis** pour répondre à la question.
    2. Si une réponse ne peut être trouvée dans les documents, répondez **clairement que vous ne savez pas**.
    3. Lorsque vous fournissez une réponse, **citez précisément les passages** ou les documents d'où vous avez extrait l'information.
    4. **Ne générez pas d'informations sensibles, inappropriées ou potentiellement incorrectes**. Limitez-vous aux informations contenues dans les documents.

    Documents fournis :
    {documents}

    Question :
    {question}

    Veuillez répondre uniquement à partir des documents ci-dessus. Si vous ne trouvez pas d'information pertinente, répondez simplement : "Je ne sais pas."
    """
    return prompt

# Exemple de requête utilisateur
# Exemple de requête utilisateur
query = "Tell me about OnePlus 6T"

# Récupération des documents pertinents
try:
    retrieved_docs = retriever.get_relevant_documents(query)
    print(f"Documents récupérés ({len(retrieved_docs)}) :")
    for doc in retrieved_docs[:3]:  # Afficher un exemple des trois premiers documents
        print(doc.page_content)
except Exception as e:
    print(f"Erreur lors de la récupération des documents : {str(e)}")
    retrieved_docs = []  # Initialisation à vide en cas d'erreur

# Vérification que des documents ont été récupérés
if retrieved_docs:
    structured_prompt = create_structured_prompt(retrieved_docs, query)
    print(f"Prompt structuré créé : {structured_prompt}")
    
    # Vérifier la configuration du pipeline avant de passer au modèle Bloom
    try:
        response = bloom_pipeline(structured_prompt)
        print(f"Réponse du modèle : {response[0]['generated_text']}")
    except Exception as e:
        print(f"Erreur lors de la génération de texte avec Bloom : {str(e)}")
else:
    print("Aucun document trouvé pour créer le prompt structuré.")
