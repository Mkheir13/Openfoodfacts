"""
Module d'utilitaires pour l'analyse et l'optimisation de OPTICS.
Fournit des fonctions pour l'entraînement, l'optimisation et la prédiction avec OPTICS.
"""
try:
	import os

	import matplotlib.pyplot as plt
	import numpy as np
	from sklearn.cluster import OPTICS
	from sklearn.metrics import (
		calinski_harabasz_score,
		davies_bouldin_score,
		silhouette_score,
	)
	from sklearn.preprocessing import StandardScaler
	import joblib
except ImportError as e:
	print(f"Erreur lors de l'importation des modules : {e}")


def find_optimal_parameters(data: np.ndarray, min_samples_range: np.ndarray | None = None, xi_range: np.ndarray | None = None, method: str = 'silhouette') -> dict[tuple[int, float], float]:
	"""
	Détermine les paramètres optimaux de OPTICS en utilisant différentes métriques.

	Paramètres:
	-----------
	data : array-like
		Les données à analyser
	min_samples_range : array-like, optional
		Plage de valeurs pour le paramètre min_samples. Si None, utilise une plage automatique
	xi_range : array-like, optional
		Plage de valeurs pour le paramètre xi. Si None, utilise une plage automatique
	method : str
		Méthode à utiliser parmi :
		- 'silhouette' : Score de silhouette (plus élevé = meilleur)
		- 'calinski_harabasz' : Score de Calinski-Harabasz (plus élevé = meilleur)
		- 'davies_bouldin' : Score de Davies-Bouldin (plus bas = meilleur)

	Retourne:
	--------
	dict : Dictionnaire contenant les scores pour chaque combinaison de paramètres
	"""
	if min_samples_range is None:
		min_samples_range = range(2, 11)

	if xi_range is None:
		xi_range = np.linspace(0.01, 0.1, 10)

	scores = {}

	for min_samples in min_samples_range:
		for xi in xi_range:
			optics = OPTICS(min_samples=min_samples, xi=xi)
			labels = optics.fit_predict(data)

			# Skip if all points are noise (-1)
			if np.all(labels == -1):
				continue

			# Skip if only one cluster (all points in same cluster)
			if len(np.unique(labels)) <= 1:
				continue

			if method == 'silhouette':
				score = silhouette_score(data, labels)
			elif method == 'calinski_harabasz':
				score = calinski_harabasz_score(data, labels)
			elif method == 'davies_bouldin':
				score = -davies_bouldin_score(data, labels)  # On inverse car plus bas = meilleur
			else:
				raise ValueError(
					"Méthode non supportée. Utilisez 'silhouette', 'calinski_harabasz', "
					"ou 'davies_bouldin'"
				)

			scores[(min_samples, xi)] = score

	return scores


def plot_parameter_scores(scores: dict[tuple[int, float], float], method: str):
	"""
	Visualise les scores pour différentes combinaisons de paramètres.

	Paramètres:
	-----------
	scores : dict
		Dictionnaire des scores pour chaque combinaison de paramètres
	method : str
		Nom de la méthode utilisée pour le titre du graphique
	"""
	# Extraction des paramètres et scores
	min_samples_values = sorted(list(set(k[0] for k in scores.keys())))
	xi_values = sorted(list(set(k[1] for k in scores.keys())))

	# Création de la matrice de scores
	score_matrix = np.zeros((len(min_samples_values), len(xi_values)))
	for (min_samples, xi), score in scores.items():
		i = min_samples_values.index(min_samples)
		j = xi_values.index(xi)
		score_matrix[i, j] = score

	# Création du graphique
	plt.figure(figsize=(12, 8))
	plt.imshow(score_matrix, aspect='auto', cmap='viridis')
	plt.colorbar(label='Score')
	plt.xlabel('xi')
	plt.ylabel('min_samples')
	plt.xticks(range(len(xi_values)), [f'{xi:.3f}' for xi in xi_values])
	plt.yticks(range(len(min_samples_values)), min_samples_values)
	plt.title(f'Analyse des paramètres optimaux - {method}')
	plt.show()


def optimize_optics(data: np.ndarray, method: str = 'grid_search', **kwargs) -> OPTICS:
	"""
	Optimise les paramètres de OPTICS en utilisant différentes stratégies.

	Paramètres:
	-----------
	data : array-like
		Les données à analyser
	method : str
		Méthode d'optimisation à utiliser :
		- 'grid_search' : Teste plusieurs combinaisons de paramètres (défaut)
		- 'auto' : Utilise les paramètres par défaut de scikit-learn
		- 'custom' : Utilise des paramètres personnalisés

	**kwargs : arguments additionnels selon la méthode
		Pour 'grid_search' :
			- min_samples_range : array-like (défaut=None)
			- xi_range : array-like (défaut=None)
			- scoring : str (défaut='silhouette')
		Pour 'auto' :
			- min_samples : int (défaut=5)
		Pour 'custom' :
			- min_samples : int
			- xi : float
			- min_cluster_size : int (défaut=None)

	Retourne:
	--------
	OPTICS : Le meilleur modèle OPTICS trouvé
	"""
	if method == 'grid_search':
		min_samples_range = kwargs.get('min_samples_range')
		xi_range = kwargs.get('xi_range')
		scoring = kwargs.get('scoring', 'silhouette')

		scores = find_optimal_parameters(
			data,
			min_samples_range=min_samples_range,
			xi_range=xi_range,
			method=scoring
		)

		if not scores:
			raise ValueError("Aucune combinaison de paramètres valide trouvée")

		# Trouver les meilleurs paramètres
		best_params = max(scores.items(), key=lambda x: x[1])[0]
		min_samples, xi = best_params

		return OPTICS(min_samples=min_samples, xi=xi)

	elif method == 'auto':
		min_samples = kwargs.get('min_samples', 5)
		return OPTICS(min_samples=min_samples)

	elif method == 'custom':
		min_samples = kwargs.get('min_samples')
		xi = kwargs.get('xi')
		min_cluster_size = kwargs.get('min_cluster_size')

		if min_samples is None or xi is None:
			raise ValueError("min_samples et xi doivent être fournis pour la méthode 'custom'")

		return OPTICS(
			min_samples=min_samples,
			xi=xi,
			min_cluster_size=min_cluster_size
		)

	raise ValueError(
		"Méthode non supportée. Utilisez 'grid_search', 'auto' ou 'custom'"
	)


def train_optics(data: np.ndarray, optimize: bool = True, save: bool = True, **kwargs) -> tuple[OPTICS, dict]:
	"""
	Fonction principale pour entraîner un modèle OPTICS.

	Paramètres:
	-----------
	data : array-like
		Les données à analyser
	optimize : bool
		Si True, détermine les paramètres optimaux
	save : bool
		Si True, sauvegarde le modèle final et le scaler
	**kwargs : arguments additionnels
		- method : str (défaut='grid_search')
			Méthode d'optimisation à utiliser
		- min_samples_range : array-like (défaut=None)
			Plage de valeurs pour min_samples
		- xi_range : array-like (défaut=None)
			Plage de valeurs pour xi
		- model_path : str (défaut='models/optics_model.pkl')
			Chemin pour sauvegarder le modèle

	Retourne:
	--------
	tuple : (OPTICS, dict)
		- Le modèle OPTICS entraîné
		- Dictionnaire contenant les informations sur l'entraînement
	"""
	# Configuration par défaut
	default_config = {
		'method': 'grid_search',
		'model_path': 'models/optics_model.pkl'
	}

	# Mise à jour de la configuration avec les kwargs
	config = default_config.copy()
	config.update(kwargs)

	# Création du dossier models si nécessaire
	if save:
		os.makedirs('models', exist_ok=True)

	# Standardisation des données
	scaler = StandardScaler()
	scaled_data = scaler.fit_transform(data)

	# Optimisation des paramètres si nécessaire
	if optimize:
		scores = find_optimal_parameters(
			scaled_data,
			min_samples_range=kwargs.get('min_samples_range'),
			xi_range=kwargs.get('xi_range'),
			method='silhouette'
		)
		plot_parameter_scores(scores, 'silhouette')

	# Création et entraînement du modèle
	model = optimize_optics(scaled_data, method=config['method'], **kwargs)
	labels = model.fit_predict(scaled_data)

	# Sauvegarde du modèle et du scaler si demandé
	if save:
		scaler_path = os.path.join(os.path.dirname(config['model_path']), 'scaler.pkl')

		save_model(model, config['model_path'])
		joblib.dump(scaler, scaler_path)
		print(f"Modèle sauvegardé dans {config['model_path']}")
		print(f"Scaler sauvegardé dans {scaler_path}")

	# Création du dictionnaire d'informations
	info = {
		'min_samples': model.min_samples,
		'xi': model.xi,
		'scaler': scaler,
		'silhouette_score': silhouette_score(scaled_data, labels),
		'calinski_harabasz_score': calinski_harabasz_score(scaled_data, labels),
		'davies_bouldin_score': davies_bouldin_score(scaled_data, labels),
		'n_clusters': len(np.unique(labels)) - (1 if -1 in labels else 0),
		'n_noise': np.sum(labels == -1)
	}

	return model, info


def predict_clusters(model: OPTICS, data: np.ndarray, scaler: StandardScaler | None = None) -> np.ndarray:
	"""
	Prédit les clusters pour de nouvelles données.

	Paramètres:
	-----------
	model : OPTICS
		Le modèle OPTICS chargé
	data : array-like
		Les données à prédire
	scaler : StandardScaler, optional
		Le scaler utilisé pour l'entraînement. Si None, les données ne seront pas standardisées

	Retourne:
	--------
	array-like
		Les labels des clusters prédits
	"""
	if scaler is not None:
		data = scaler.transform(data)
	return model.fit_predict(data)


def save_model(model: OPTICS, path: str = "models/optics_model.pkl"):
	"""
	Sauvegarde le modèle OPTICS entraîné.

	Paramètres:
	-----------
	model : OPTICS
		Le modèle à sauvegarder
	path : str
		Chemin où sauvegarder le modèle
	"""
	joblib.dump(model, path)


def load_optics(model_path: str, scaler_path: str | None = None) -> tuple[OPTICS, StandardScaler | None]:
	"""
	Charge un modèle OPTICS pré-entraîné et son scaler associé.

	Paramètres:
	-----------
	model_path : str
		Chemin vers le fichier du modèle OPTICS sauvegardé
	scaler_path : str, optional
		Chemin vers le fichier du scaler sauvegardé. Si None, cherchera dans le même dossier

	Retourne:
	--------
	tuple : (OPTICS, StandardScaler)
		- Le modèle OPTICS chargé
		- Le scaler associé (si trouvé)

	Raises:
	------
	FileNotFoundError
		Si le fichier du modèle n'existe pas
	"""
	if not os.path.exists(model_path):
		raise FileNotFoundError(f"Le fichier du modèle n'existe pas : {model_path}")

	# Chargement du modèle
	model = joblib.load(model_path)

	# Chargement du scaler
	scaler = None
	if scaler_path is None:
		# Cherche le scaler dans le même dossier que le modèle
		scaler_path = os.path.join(os.path.dirname(model_path), 'scaler.pkl')

	if os.path.exists(scaler_path):
		scaler = joblib.load(scaler_path)

	return model, scaler
