import numpy as np
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

def silhouette_metric(X: np.ndarray, labels: np.ndarray) -> float:
	"""
	Calculate the silhouette score for clustering evaluation
	Higher is better (range: -1 to 1)
	"""
	return silhouette_score(X, labels)

def calinski_harabasz_metric(X: np.ndarray, labels: np.ndarray) -> float:
	"""
	Calculate the Calinski-Harabasz score for clustering evaluation
	Higher is better
	"""
	return calinski_harabasz_score(X, labels)

def davies_bouldin_metric(X: np.ndarray, labels: np.ndarray) -> float:
	"""
	Calculate the Davies-Bouldin score for clustering evaluation
	Lower is better (minimum is 0)
	"""
	return davies_bouldin_score(X, labels)

def compare_clustering_metrics(X: np.ndarray, labels: np.ndarray) -> dict[str, float]:
	"""
	Compare all clustering metrics and return them in a dictionary
	"""
	metrics = {
		'silhouette': silhouette_metric(X, labels),
		'calinski_harabasz': calinski_harabasz_metric(X, labels),
		'davies_bouldin': davies_bouldin_metric(X, labels)
	}

	print("Résultats des métriques de clustering :")
	print(f"Score de Silhouette : {metrics['silhouette']:.4f}")
	print(f"Score de Calinski-Harabasz : {metrics['calinski_harabasz']:.4f}")
	print(f"Score de Davies-Bouldin : {metrics['davies_bouldin']:.4f}")
	print(f"Score d'Inertie : {metrics['inertia']:.4f}")

	return metrics
