from DataLoad import charger_donnees
from langchain.text_splitter import RecursiveCharacterTextSplitter

def decouper_textes(fichier):
    """
    Charge un fichier JSON Lines et découpe les textes en morceaux plus petits.

    Args:
        fichier (str): Chemin vers le fichier JSON Lines.

    Returns:
        list: Liste des textes découpés.
    """
    # Charger les données
    data = charger_donnees(fichier)
    
    # Initialiser le découpeur de texte
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=128)
    
    # Découper les textes
    split_texts = []
    for product_text in data:
        split_texts.extend(text_splitter.split_text(product_text))
    
    return split_texts
