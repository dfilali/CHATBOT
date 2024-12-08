import jsonlines

def charger_donnees(fichier):
    """
    Charge et transforme les données d'un fichier JSON Lines.

    Args:
        fichier (str): Chemin vers le fichier JSON Lines.

    Returns:
        list: Liste des textes combinés.
    """
    data = []
    with jsonlines.open(fichier) as reader:
        for obj in reader:
            description = obj.get('description', [])
            features = obj.get('features', [])
            title = obj.get('title', "")
            combined_text = f"{title}. Features: {' '.join(features)}. Description: {' '.join(description)}"
            data.append(combined_text)
    return data
print(charger_donnees("meta.jsonl"))