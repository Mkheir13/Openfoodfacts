"""
Pipeline principal pour le clustering des données OpenFoodFacts.
"""
import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, Any

# Ajout du dossier parent au PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from scripts.data.loading import load_openfoodfacts
from scripts.models.clustering import prepare_features_for_clustering
from scripts.models.model_comparison import ModelComparison

def run_clustering_pipeline(
	n_samples: int = 10000,
	max_categories: int = 30,
	min_unique_ratio: float = 0.05,
	random_state: int = 42,
	save_results: bool = True
) -> Dict[str, Any]:
	"""
	Exécute la pipeline complète de clustering.
	
	Args:
		n_samples: Nombre d'échantillons à charger
		max_categories: Nombre maximum de catégories pour les variables catégorielles
		min_unique_ratio: Ratio minimum de valeurs uniques pour les variables numériques
		random_state: État aléatoire pour la reproductibilité
		save_results: Si True, sauvegarde les résultats
	
	Returns:
		Dict contenant les résultats de l'analyse
	"""
	print("1. Chargement des données...")
	df = load_openfoodfacts(n_samples=n_samples)
	
	print("\n2. Préparation des features pour le clustering...")
	df_clean = prepare_features_for_clustering(
		df,
		max_categories=max_categories,
		min_unique_ratio=min_unique_ratio
	)
	
	print("\n3. Conversion en array numpy...")
	X = df_clean.to_numpy()
	
	print("\n4. Initialisation de la comparaison des modèles...")
	comparison = ModelComparison(X, random_state=random_state)
	
	print("\n5. Entraînement de tous les modèles...")
	results = comparison.train_all()
	
	print("\n6. Comparaison des métriques...")
	metrics_df = comparison.compare_metrics()
	print("\nMétriques de comparaison:")
	print(metrics_df)
	
	if save_results:
		print("\n7. Sauvegarde des résultats...")
		os.makedirs('results', exist_ok=True)
		comparison.save_results('results')
		metrics_df.to_csv('results/metrics_comparison.csv', index=False)
	
	return {
		'data': df_clean,
		'comparison': comparison,
		'results': results,
		'metrics': metrics_df
	}

if __name__ == "__main__":
	# Exécution de la pipeline
	results = run_clustering_pipeline()
	
	# Affichage des visualisations
	results['comparison'].plot_comparison()
	results['comparison'].plot_clusters() 