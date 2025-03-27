"""
Script utilitaire pour prétraiter et échantillonner les fichiers volumineux.
Ce script peut être exécuté en dehors de Streamlit pour créer des versions plus petites
des fichiers trop volumineux pour être chargés directement dans l'application.
"""

import os
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm


def count_lines(file_path):
    """Compte le nombre de lignes dans un fichier."""
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        return sum(1 for _ in tqdm(f, desc="Comptage des lignes"))


def sample_file(input_path, output_path, sample_size=10000, random_state=42):
    """
    Échantillonne un fichier CSV volumineux.
    
    Args:
        input_path: Chemin vers le fichier CSV d'entrée
        output_path: Chemin où sauvegarder le fichier échantillonné
        sample_size: Nombre de lignes à extraire
        random_state: Seed pour la reproductibilité
    """
    # Fixer la graine aléatoire
    np.random.seed(random_state)
    
    # Compter le nombre de lignes dans le fichier
    print(f"Analyse du fichier {input_path}...")
    total_rows = count_lines(input_path)
    print(f"Le fichier contient {total_rows} lignes.")
    
    if total_rows <= sample_size:
        print("Le fichier a moins de lignes que la taille d'échantillon demandée.")
        print(f"Copie du fichier entier vers {output_path}...")
        df = pd.read_csv(input_path)
        df.to_csv(output_path, index=False)
        print("Copie terminée.")
        return
    
    # Calculer la probabilité de sélection de chaque ligne
    probability = sample_size / total_rows
    
    # Échantillonnage aléatoire du fichier
    print(f"Échantillonnage de {sample_size} lignes sur {total_rows}...")
    
    # Lire l'en-tête (première ligne)
    with open(input_path, 'r', encoding='utf-8', errors='replace') as infile:
        header = infile.readline().strip()
    
    # Écrire l'en-tête dans le fichier de sortie
    with open(output_path, 'w', encoding='utf-8') as outfile:
        outfile.write(header + '\n')
    
    # Échantillonnage des lignes
    sampled_count = 0
    with open(input_path, 'r', encoding='utf-8', errors='replace') as infile:
        # Skip the header
        next(infile)
        
        with open(output_path, 'a', encoding='utf-8') as outfile:
            for line in tqdm(infile, total=total_rows-1, desc="Échantillonnage"):
                if np.random.random() < probability and sampled_count < sample_size:
                    outfile.write(line)
                    sampled_count += 1
    
    print(f"Échantillonnage terminé. {sampled_count} lignes écrites dans {output_path}")


def get_reduced_file(input_path, output_dir="data/reduced", sample_size=10000):
    """
    Crée une version réduite d'un fichier s'il n'existe pas déjà.
    
    Args:
        input_path: Chemin du fichier d'entrée
        output_dir: Dossier de sortie
        sample_size: Taille de l'échantillon
        
    Returns:
        Le chemin du fichier réduit
    """
    # Créer le dossier de sortie s'il n'existe pas
    os.makedirs(output_dir, exist_ok=True)
    
    # Construire le nom du fichier de sortie
    filename = os.path.basename(input_path)
    name, ext = os.path.splitext(filename)
    output_filename = f"{name}_{sample_size}{ext}"
    output_path = os.path.join(output_dir, output_filename)
    
    # Vérifier si le fichier existe déjà
    if os.path.exists(output_path):
        print(f"Le fichier réduit {output_path} existe déjà.")
        return output_path
    
    # Créer le fichier réduit
    sample_file(input_path, output_path, sample_size)
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Échantillonnage de fichiers CSV volumineux")
    parser.add_argument("input_file", help="Chemin vers le fichier CSV d'entrée")
    parser.add_argument("--output", "-o", help="Chemin où sauvegarder le fichier échantillonné (optionnel)")
    parser.add_argument("--size", "-s", type=int, default=10000, help="Nombre de lignes à extraire (défaut: 10000)")
    parser.add_argument("--seed", type=int, default=42, help="Graine aléatoire (défaut: 42)")
    
    args = parser.parse_args()
    
    # Si l'utilisateur n'a pas spécifié de chemin de sortie, en créer un par défaut
    if not args.output:
        output_dir = "data/reduced"
        os.makedirs(output_dir, exist_ok=True)
        
        filename = os.path.basename(args.input_file)
        name, ext = os.path.splitext(filename)
        output_filename = f"{name}_{args.size}{ext}"
        output_path = os.path.join(output_dir, output_filename)
    else:
        output_path = args.output
    
    sample_file(args.input_file, output_path, args.size, args.seed) 