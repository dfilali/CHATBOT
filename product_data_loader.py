import logging
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO)

def load_product_data(file_path='meta.jsonl'):
    """Charge les données des produits depuis un fichier JSONL."""
    try:
        product_data = pd.read_json(file_path, lines=True)
        logging.info("Product data loaded successfully.")
        return product_data
    except Exception as e:
        logging.error(f"Error loading product data: {e}")
        raise

def process_descriptions(product_data):
    """Traitement des descriptions des produits pour les convertir en une liste de chaînes de texte."""
    descriptions = product_data['description'].apply(
        lambda x: " ".join(x) if isinstance(x, list) else str(x)
    ).fillna("").tolist()
    return descriptions
