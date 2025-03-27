"""
Module de comparaison des différents modèles de clustering.
Permet d'évaluer et de comparer les performances de K-means, GMM, DBSCAN et OPTICS.
"""
try:
	import os
	import time
	from typing import Dict, List, Tuple, Union

	import matplotlib.pyplot as plt
	import numpy as np
	import pandas as pd
	from sklearn.cluster import DBSCAN, KMeans, OPTICS
	from sklearn.mixture import GaussianMixture
	from sklearn.metrics import (
		calinski_harabasz_score,
		davies_bouldin_score,
		silhouette_score,
	)
	from sklearn.preprocessing import StandardScaler
	import seaborn as sns
except ImportError as e:
	print(f"Erreur lors de l'importation des modules : {e}")


class ModelComparison:
	"""
	Classe pour comparer différents modèles de clustering.
	"""

	def __init__(self, data: np.ndarray, random_state: int = 42):
		"""
		Initialise la comparaison des modèles.
		
		Paramètres:
		-----------
		data : array-like
			Les données à analyser
		random_state : int
			État aléatoire pour la reproductibilité
		"""
		self.data = data
		self.random_state = random_state
		self.scaler = StandardScaler()
		self.scaled_data = self.scaler.fit_transform(data)
		self.results = {}
		self.models = {
			'kmeans': None,
			'gmm': None,
			'dbscan': None,
			'optics': None
		}

	def train_kmeans(self, n_clusters: int = None, **kwargs) -> Dict:
		"""
		Entraîne un modèle K-means.
		
		Paramètres:
		-----------
		n_clusters : int, optional
			Nombre de clusters. Si None, sera déterminé automatiquement
		**kwargs : arguments additionnels pour K-means
		
		Retourne:
		--------
		dict : Informations sur l'entraînement
		"""
		from kmeans_training import train_kmeans
		
		start_time = time.time()
		model, info = train_kmeans(
			self.scaled_data,
			n_clusters=n_clusters,
			**kwargs
		)
		training_time = time.time() - start_time
		
		self.models['kmeans'] = model
		info['training_time'] = training_time
		self.results['kmeans'] = info
		
		return info

	def train_gmm(self, n_components: int = None, **kwargs) -> Dict:
		"""
		Entraîne un modèle GMM.
		
		Paramètres:
		-----------
		n_components : int, optional
			Nombre de composantes. Si None, sera déterminé automatiquement
		**kwargs : arguments additionnels pour GMM
		
		Retourne:
		--------
		dict : Informations sur l'entraînement
		"""
		from gmm_training import train_gmm
		
		start_time = time.time()
		model, info = train_gmm(
			self.scaled_data,
			n_components=n_components,
			**kwargs
		)
		training_time = time.time() - start_time
		
		self.models['gmm'] = model
		info['training_time'] = training_time
		self.results['gmm'] = info
		
		return info

	def train_dbscan(self, **kwargs) -> Dict:
		"""
		Entraîne un modèle DBSCAN.
		
		Paramètres:
		-----------
		**kwargs : arguments pour DBSCAN
		
		Retourne:
		--------
		dict : Informations sur l'entraînement
		"""
		from dbscan_training import train_dbscan
		
		start_time = time.time()
		model, info = train_dbscan(
			self.scaled_data,
			**kwargs
		)
		training_time = time.time() - start_time
		
		self.models['dbscan'] = model
		info['training_time'] = training_time
		self.results['dbscan'] = info
		
		return info

	def train_optics(self, **kwargs) -> Dict:
		"""
		Entraîne un modèle OPTICS.
		
		Paramètres:
		-----------
		**kwargs : arguments pour OPTICS
		
		Retourne:
		--------
		dict : Informations sur l'entraînement
		"""
		from optics_training import train_optics
		
		start_time = time.time()
		model, info = train_optics(
			self.scaled_data,
			**kwargs
		)
		training_time = time.time() - start_time
		
		self.models['optics'] = model
		info['training_time'] = training_time
		self.results['optics'] = info
		
		return info

	def train_all(self, **kwargs) -> Dict:
		"""
		Entraîne tous les modèles.
		
		Paramètres:
		-----------
		**kwargs : arguments pour les différents modèles
		
		Retourne:
		--------
		dict : Informations sur l'entraînement de tous les modèles
		"""
		self.train_kmeans(**kwargs.get('kmeans', {}))
		self.train_gmm(**kwargs.get('gmm', {}))
		self.train_dbscan(**kwargs.get('dbscan', {}))
		self.train_optics(**kwargs.get('optics', {}))
		
		return self.results

	def compare_metrics(self) -> pd.DataFrame:
		"""
		Compare les métriques entre les différents modèles.
		
		Retourne:
		--------
		pd.DataFrame : Tableau comparatif des métriques
		"""
		metrics = []
		for model_name, info in self.results.items():
			metrics.append({
				'Modèle': model_name,
				'Temps d\'entraînement (s)': info['training_time'],
				'Score de silhouette': info['silhouette_score'],
				'Score de Calinski-Harabasz': info['calinski_harabasz_score'],
				'Score de Davies-Bouldin': info['davies_bouldin_score'],
				'Nombre de clusters': info.get('n_clusters', info.get('n_components', None)),
				'Nombre de points de bruit': info.get('n_noise', 0)
			})
		
		return pd.DataFrame(metrics)

	def plot_comparison(self, metric: str = 'silhouette_score'):
		"""
		Visualise la comparaison des modèles selon une métrique.
		
		Paramètres:
		-----------
		metric : str
			Métrique à visualiser
		"""
		df = self.compare_metrics()
		
		plt.figure(figsize=(10, 6))
		sns.barplot(data=df, x='Modèle', y=metric)
		plt.title(f'Comparaison des modèles - {metric}')
		plt.xticks(rotation=45)
		plt.tight_layout()
		plt.show()

	def plot_clusters(self, n_cols: int = 2):
		"""
		Visualise les clusters pour chaque modèle.
		
		Paramètres:
		-----------
		n_cols : int
			Nombre de colonnes dans la grille de visualisation
		"""
		n_models = len(self.models)
		n_rows = (n_models + n_cols - 1) // n_cols
		
		plt.figure(figsize=(15, 5 * n_rows))
		
		for i, (model_name, model) in enumerate(self.models.items(), 1):
			if model is None:
				continue
				
			plt.subplot(n_rows, n_cols, i)
			labels = model.fit_predict(self.scaled_data)
			
			# Si les données sont 2D, on peut les visualiser directement
			if self.scaled_data.shape[1] == 2:
				plt.scatter(
					self.scaled_data[:, 0],
					self.scaled_data[:, 1],
					c=labels,
					cmap='viridis'
				)
			else:
				# Sinon, on utilise PCA pour réduire à 2D
				from sklearn.decomposition import PCA
				pca = PCA(n_components=2)
				data_2d = pca.fit_transform(self.scaled_data)
				plt.scatter(
					data_2d[:, 0],
					data_2d[:, 1],
					c=labels,
					cmap='viridis'
				)
			
			plt.title(f'Clusters - {model_name}')
			plt.xlabel('Première composante')
			plt.ylabel('Deuxième composante')
		
		plt.tight_layout()
		plt.show()

	def save_results(self, path: str = 'results'):
		"""
		Sauvegarde les résultats de la comparaison.
		
		Paramètres:
		-----------
		path : str
			Chemin où sauvegarder les résultats
		"""
		os.makedirs(path, exist_ok=True)
		
		# Sauvegarde des métriques
		metrics_df = self.compare_metrics()
		metrics_df.to_csv(os.path.join(path, 'metrics.csv'), index=False)
		
		# Sauvegarde des modèles
		for model_name, model in self.models.items():
			if model is not None:
				joblib.dump(model, os.path.join(path, f'{model_name}_model.pkl'))
		
		# Sauvegarde du scaler
		joblib.dump(self.scaler, os.path.join(path, 'scaler.pkl'))
		
		print(f"Résultats sauvegardés dans {path}")

	def load_results(self, path: str = 'results'):
		"""
		Charge les résultats d'une comparaison précédente.
		
		Paramètres:
		-----------
		path : str
			Chemin où charger les résultats
		"""
		# Chargement des métriques
		metrics_df = pd.read_csv(os.path.join(path, 'metrics.csv'))
		
		# Chargement des modèles
		for model_name in self.models.keys():
			model_path = os.path.join(path, f'{model_name}_model.pkl')
			if os.path.exists(model_path):
				self.models[model_name] = joblib.load(model_path)
		
		# Chargement du scaler
		scaler_path = os.path.join(path, 'scaler.pkl')
		if os.path.exists(scaler_path):
			self.scaler = joblib.load(scaler_path)
			self.scaled_data = self.scaler.transform(self.data)
		
		print(f"Résultats chargés depuis {path}")


def compare_clustering_models(
	data: np.ndarray,
	save_results: bool = True,
	**kwargs
) -> ModelComparison:
	"""
	Fonction utilitaire pour comparer rapidement les modèles de clustering.
	
	Paramètres:
	-----------
	data : array-like
		Les données à analyser
	save_results : bool
		Si True, sauvegarde les résultats
	**kwargs : arguments pour les différents modèles
	
	Retourne:
	--------
	ModelComparison : Instance de la classe de comparaison
	"""
	comparison = ModelComparison(data)
	comparison.train_all(**kwargs)
	
	if save_results:
		comparison.save_results()
	
	return comparison 