from Splitter import decouper_textes
from EmbeddingGeneration import generer_embeddings

# Chemin vers le fichier JSON Lines
fichier = 'meta.jsonl'

# Étape 1 : Charger et découper les textes
split_texts = decouper_textes(fichier)

# Étape 2 : Générer les embeddings
embeddings = generer_embeddings(split_texts)

# Étape 3 : Afficher un aperçu des résultats
for i, emb in enumerate(embeddings[:5]):
    print(f"Texte segmenté {i+1} - Embedding : {emb[:5]}")  # Afficher les 5 premières valeurs de chaque embedding
