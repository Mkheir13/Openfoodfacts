"""
Utilitaire pour créer une version réduite du fichier openfoodfacts_100000.csv
permettant de l'utiliser dans l'application Streamlit.
"""

import os
import sys
from scripts.data.sample_large_files import get_reduced_file

# Taille d'échantillon par défaut
SAMPLE_SIZE = 10000

def main():
    """Fonction principale pour créer l'échantillon."""
    # Vérifier les arguments
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        # Chercher le fichier openfoodfacts_100000.csv s'il existe
        input_file = ""
        possible_paths = [
            "data/openfoodfacts_100000.csv",
            "openfoodfacts_100000.csv",
            "../data/openfoodfacts_100000.csv"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                input_file = path
                break
        
        if not input_file:
            print("Erreur: Veuillez spécifier le chemin du fichier CSV à traiter.")
            print("Usage: python create_reduced_file.py [chemin_fichier.csv] [taille_échantillon]")
            return
    
    # Vérifier la taille d'échantillon
    sample_size = SAMPLE_SIZE
    if len(sys.argv) > 2:
        try:
            sample_size = int(sys.argv[2])
        except ValueError:
            print(f"La taille d'échantillon '{sys.argv[2]}' n'est pas valide. Utilisation de {SAMPLE_SIZE}.")
    
    # Créer le dossier reduced s'il n'existe pas déjà
    output_dir = "data/reduced"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Création d'un échantillon de {sample_size} lignes à partir de {input_file}...")
    
    # Obtenir le fichier réduit
    reduced_file = get_reduced_file(input_file, output_dir, sample_size)
    
    print(f"Échantillon créé avec succès: {reduced_file}")
    print("Vous pouvez maintenant lancer l'application Streamlit avec:")
    print("streamlit run app.py")

if __name__ == "__main__":
    main() 