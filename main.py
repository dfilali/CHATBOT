import streamlit as st
from model import generate_response

# Titre de l'application Streamlit
st.title("Product Information Query")

# Demande de la requête utilisateur
query = st.text_input("Enter your query:")

# Paramètres pour ajuster la génération du texte
temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
top_p = st.slider("Top-p", min_value=0.0, max_value=1.0, value=0.9, step=0.1)

# Si une requête est fournie, générer et afficher la réponse
if query:
    response = generate_response(query, temperature=temperature, top_p=top_p)
    st.write("**Generated Response:**")
    st.write(response)
