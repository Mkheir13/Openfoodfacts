"""
Module d'utilitaires pour l'analyse et l'optimisation des Gaussian Mixture Models.
Fournit des fonctions pour l'entraînement, l'optimisation et la prédiction avec GMM.
"""
try:
	import os

	import matplotlib.pyplot as plt
	import numpy as np
	from sklearn.mixture import GaussianMixture
	from sklearn.metrics import (
		calinski_harabasz_score,
		davies_bouldin_score,
		silhouette_score,
	)
	from sklearn.model_selection import GridSearchCV
	from sklearn.preprocessing import StandardScaler
	import joblib
except ImportError as e:
	print(f"Erreur lors de l'importation des modules : {e}")


def find_optimal_components(data, n_max=10, n_min=2, method='silhouette'):
	"""
	Détermine le nombre optimal de composantes en utilisant différentes métriques.
	
	Paramètres:
	-----------
	data : array-like
		Les données à analyser
	n_max : int
		Nombre maximum de composantes à tester
	n_min : int
		Nombre minimum de composantes à tester
	method : str
		Méthode à utiliser parmi :
		- 'silhouette' : Score de silhouette (plus élevé = meilleur)
		- 'calinski_harabasz' : Score de Calinski-Harabasz (plus élevé = meilleur)
		- 'davies_bouldin' : Score de Davies-Bouldin (plus bas = meilleur)
		- 'bic' : BIC (plus bas = meilleur)
	
	Retourne:
	--------
	dict : Dictionnaire contenant les scores pour chaque nombre de composantes
	"""
	scores = {}

	for n in range(n_min, n_max + 1):
		gmm = GaussianMixture(n_components=n, random_state=42)
		labels = gmm.fit_predict(data)

		if method == 'silhouette':
			score = silhouette_score(data, labels)
		elif method == 'calinski_harabasz':
			score = calinski_harabasz_score(data, labels)
		elif method == 'davies_bouldin':
			score = -davies_bouldin_score(data, labels)  # On inverse car plus bas = meilleur
		elif method == 'bic':
			score = -gmm.bic(data)  # On inverse car plus bas = meilleur
		else:
			raise ValueError(
				"Méthode non supportée. Utilisez 'silhouette', 'calinski_harabasz', "
				"'davies_bouldin' ou 'bic'"
			)

		scores[n] = score

	return scores


def plot_component_scores(scores, method):
	"""
	Visualise les scores pour différents nombres de composantes.
	
	Paramètres:
	-----------
	scores : dict
		Dictionnaire des scores pour chaque nombre de composantes
	method : str
		Nom de la méthode utilisée pour le titre du graphique
	"""
	plt.figure(figsize=(10, 6))
	plt.plot(list(scores.keys()), list(scores.values()), marker='o')
	plt.xlabel('Nombre de composantes')
	plt.ylabel('Score')
	plt.title(f'Analyse du nombre optimal de composantes - {method}')
	plt.grid(True)
	plt.show()


def optimize_gmm(data, n_components, method='multiple_init', **kwargs):
	"""
	Optimise les paramètres de GMM en utilisant différentes stratégies.
	
	Paramètres:
	-----------
	data : array-like
		Les données à analyser
	n_components : int
		Nombre de composantes souhaité
	method : str
		Méthode d'optimisation à utiliser :
		- 'multiple_init' : Teste plusieurs initialisations (défaut)
		- 'grid_search' : Utilise GridSearchCV pour tester plusieurs combinaisons
		- 'covariance_type' : Compare différents types de matrices de covariance
		- 'custom_init' : Utilise des paramètres initiaux personnalisés
	
	**kwargs : arguments additionnels selon la méthode
		Pour 'multiple_init' :
			- n_init : int (défaut=10)
			- max_iter : int (défaut=100)
		Pour 'grid_search' :
			- param_grid : dict (défaut=None)
			- cv : int (défaut=3)
		Pour 'covariance_type' :
			- types : list (défaut=['full', 'tied', 'diag', 'spherical'])
	
	Retourne:
	--------
	GaussianMixture : Le meilleur modèle GMM trouvé
	"""
	# Configuration par défaut pour chaque méthode
	default_configs = {
		'multiple_init': {
			'n_init': 10,
			'max_iter': 100,
			'init_params': 'kmeans'
		},
		'grid_search': {
			'param_grid': {
				'init_params': ['kmeans', 'random'],
				'n_init': [10, 20],
				'max_iter': [100, 200],
				'covariance_type': ['full', 'tied', 'diag', 'spherical']
			},
			'cv': 3
		},
		'covariance_type': {
			'types': ['full', 'tied', 'diag', 'spherical']
		}
	}

	# Mise à jour des configurations avec les kwargs
	config = default_configs.get(method, {})
	config.update(kwargs)

	if method == 'multiple_init':
		best_model = None
		best_score = -np.inf

		for _ in range(config['n_init']):
			gmm = GaussianMixture(
				n_components=n_components,
				init_params=config['init_params'],
				max_iter=config['max_iter'],
				random_state=None
			)
			gmm.fit(data)

			score = silhouette_score(data, gmm.predict(data))

			if score > best_score:
				best_score = score
				best_model = gmm

		return best_model

	if method == 'grid_search':
		gmm = GaussianMixture(n_components=n_components, random_state=42)
		grid_search = GridSearchCV(
			gmm,
			param_grid=config['param_grid'],
			cv=config['cv'],
			scoring='silhouette'
		)
		grid_search.fit(data)

		return grid_search.best_estimator_

	if method == 'covariance_type':
		best_model = None
		best_score = -np.inf

		for cov_type in config['types']:
			gmm = GaussianMixture(
				n_components=n_components,
				covariance_type=cov_type,
				random_state=42
			)
			gmm.fit(data)

			score = silhouette_score(data, gmm.predict(data))

			if score > best_score:
				best_score = score
				best_model = gmm

		return best_model

	if method == 'custom_init':
		init_params = kwargs.get('init_params')
		if init_params is None:
			raise ValueError("init_params doit être fourni pour la méthode 'custom_init'")

		gmm = GaussianMixture(
			n_components=n_components,
			init_params=init_params,
			n_init=1,
			random_state=42
		)
		gmm.fit(data)

		return gmm

	raise ValueError(
		"Méthode non supportée. Utilisez 'multiple_init', 'grid_search', "
		"'covariance_type' ou 'custom_init'"
	)


def train_gmm(data, n_components=None, optimize=True, save=True, **kwargs):
	"""
	Fonction principale pour entraîner un modèle GMM.
	
	Paramètres:
	-----------
	data : array-like
		Les données à analyser
	n_components : int, optional
		Nombre de composantes souhaité. Si None, sera déterminé automatiquement
	optimize : bool
		Si True, détermine le nombre optimal de composantes
	save : bool
		Si True, sauvegarde le modèle final et le scaler
	**kwargs : arguments additionnels
		- method : str (défaut='silhouette')
			Méthode pour déterminer le nombre optimal de composantes
		- n_max : int (défaut=10)
			Nombre maximum de composantes à tester
		- n_min : int (défaut=2)
			Nombre minimum de composantes à tester
		- optimize_method : str (défaut='multiple_init')
			Méthode d'optimisation des paramètres
		- model_path : str (défaut='models/gmm_model.pkl')
			Chemin pour sauvegarder le modèle
	
	Retourne:
	--------
	tuple : (GaussianMixture, dict)
		- Le modèle GMM entraîné
		- Dictionnaire contenant les informations sur l'entraînement
	"""
	# Configuration par défaut
	default_config = {
		'method': 'silhouette',
		'n_max': 10,
		'n_min': 2,
		'optimize_method': 'multiple_init',
		'model_path': 'models/gmm_model.pkl'
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

	# Détermination du nombre optimal de composantes si nécessaire
	if optimize and n_components is None:
		scores = find_optimal_components(
			scaled_data,
			n_max=config['n_max'],
			n_min=config['n_min'],
			method=config['method']
		)
		plot_component_scores(scores, config['method'])

		# Sélection du nombre optimal de composantes
		n_components = max(scores.items(), key=lambda x: x[1])[0]
		print(f"Nombre optimal de composantes trouvé : {n_components}")

	# Optimisation des paramètres
	model = optimize_gmm(
		scaled_data,
		n_components=n_components,
		method=config['optimize_method'],
		**kwargs
	)

	# Sauvegarde du modèle et du scaler si demandé
	if save:
		scaler_path = os.path.join(os.path.dirname(config['model_path']), 'scaler.pkl')

		save_model(model, config['model_path'])
		joblib.dump(scaler, scaler_path)
		print(f"Modèle sauvegardé dans {config['model_path']}")
		print(f"Scaler sauvegardé dans {scaler_path}")

	# Création du dictionnaire d'informations
	info = {
		'n_components': n_components,
		'optimize_method': config['optimize_method'],
		'scaler': scaler,
		'silhouette_score': silhouette_score(scaled_data, model.predict(scaled_data)),
		'calinski_harabasz_score': calinski_harabasz_score(scaled_data, model.predict(scaled_data)),
		'davies_bouldin_score': davies_bouldin_score(scaled_data, model.predict(scaled_data)),
		'bic': model.bic(scaled_data),
		'aic': model.aic(scaled_data)
	}

	return model, info


def predict_clusters(model, data, scaler=None):
	"""
	Prédit les clusters pour de nouvelles données.
	
	Paramètres:
	-----------
	model : GaussianMixture
		Le modèle GMM chargé
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
	return model.predict(data)


def save_model(model, path="models/gmm_model.pkl"):
	"""
	Sauvegarde le modèle GMM entraîné.
	
	Paramètres:
	-----------
	model : GaussianMixture
		Le modèle à sauvegarder
	path : str
		Chemin où sauvegarder le modèle
	"""
	joblib.dump(model, path)


def load_gmm(model_path, scaler_path=None):
	"""
	Charge un modèle GMM pré-entraîné et son scaler associé.
	
	Paramètres:
	-----------
	model_path : str
		Chemin vers le fichier du modèle GMM sauvegardé
	scaler_path : str, optional
		Chemin vers le fichier du scaler sauvegardé. Si None, cherchera dans le même dossier
	
	Retourne:
	--------
	tuple : (GaussianMixture, StandardScaler)
		- Le modèle GMM chargé
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
