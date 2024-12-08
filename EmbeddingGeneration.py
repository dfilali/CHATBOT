import requests
import json
import asyncio

async def get_embeddings_async(session, text, model="llama3:latest", url="http://localhost:11434/api/v1/run"):
    """
    Obtient les embeddings de manière asynchrone.

    Args:
        session: Une session de requête HTTP.
        text (str): Le texte.
        model (str, optional): Nom du modèle. Defaults to "llama3:latest".
        url (str, optional): URL de l'API. Defaults to "http://localhost:11434/api/v1/run".

    Returns:
        list: Liste des embeddings.
    """
    # ... (code similaire à la fonction originale, avec gestion des erreurs améliorée)

async def generer_embeddings_async(split_texts, model="llama3:latest", url="http://localhost:11434/api/v1/run"):
    """
    Génère les embeddings de manière asynchrone pour une liste de textes.

    Args:
        split_texts (list): Liste des textes.
        model (str, optional): Nom du modèle. Defaults to "llama3:latest".
        url (str, optional): URL de l'API. Defaults to "http://localhost:11434/api/v1/run".

    Returns:
        list: Liste des embeddings.
    """
    async with aiohttp.ClientSession() as session:
        tasks = [get_embeddings_async(session, text, model, url) for text in split_texts]
        return await asyncio.gather(*tasks)

# Utilisation :
asyncio.run(generer_embeddings_async(split_texts))