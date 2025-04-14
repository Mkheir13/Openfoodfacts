"""
Module d'utilitaires pour l'analyse et l'optimisation de DBSCAN.
Fournit des fonctions pour l'entraînement, l'optimisation et la prédiction avec DBSCAN.
"""
try:
    import os

    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.cluster import DBSCAN
    from sklearn.metrics import (
        calinski_harabasz_score,
        davies_bouldin_score,
        silhouette_score,
    )
    from sklearn.preprocessing import StandardScaler
    import joblib
except ImportError as e:
    print(f"Erreur lors de l'importation des modules : {e}")


def find_optimal_parameters(
    data: np.ndarray,
    eps_range: np.ndarray | None = None,
    min_samples_range: np.ndarray | None = None,
    method: str = "silhouette",
) -> dict[tuple[float, int], float]:
    """
    Détermine les paramètres optimaux de DBSCAN en utilisant différentes métriques.

    Paramètres:
    -----------
    data : array-like
        Les données à analyser
    eps_range : array-like, optional
        Plage de valeurs pour le paramètre eps. Si None, utilise une plage automatique
    min_samples_range : array-like, optional
        Plage de valeurs pour le paramètre min_samples. Si None, utilise une plage automatique
    method : str
        Méthode à utiliser parmi :
        - 'silhouette' : Score de silhouette (plus élevé = meilleur)
        - 'calinski_harabasz' : Score de Calinski-Harabasz (plus élevé = meilleur)
        - 'davies_bouldin' : Score de Davies-Bouldin (plus bas = meilleur)

    Retourne:
    --------
    dict : Dictionnaire contenant les scores pour chaque combinaison de paramètres
    """
    if eps_range is None:
        # Calcul automatique de la plage eps basée sur les distances entre points
        from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=2).fit(data)
        distances, _ = nbrs.kneighbors(data)
        eps_range = np.linspace(distances.mean() * 0.5, distances.mean() * 2, 10)

    if min_samples_range is None:
        min_samples_range = range(2, 11)

    scores: dict[tuple[float, int], float] = {}

    for eps in eps_range:
        for min_samples in min_samples_range:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(data)

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

            scores[(eps, min_samples)] = score

    return scores


def plot_parameter_scores(scores: dict[tuple[float, int], float], method: str):
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
    eps_values = sorted(list(set(k[0] for k in scores.keys())))
    min_samples_values = sorted(list(set(k[1] for k in scores.keys())))

    # Création de la matrice de scores
    score_matrix = np.zeros((len(eps_values), len(min_samples_values)))
    for (eps, min_samples), score in scores.items():
        i = eps_values.index(eps)
        j = min_samples_values.index(min_samples)
        score_matrix[i, j] = score

    # Création du graphique
    plt.figure(figsize=(12, 8))
    plt.imshow(score_matrix, aspect='auto', cmap='viridis')
    plt.colorbar(label='Score')
    plt.xlabel('min_samples')
    plt.ylabel('eps')
    plt.xticks(range(len(min_samples_values)), min_samples_values)
    plt.yticks(range(len(eps_values)), [f'{eps:.3f}' for eps in eps_values])
    plt.title(f'Analyse des paramètres optimaux - {method}')
    plt.show()


def optimize_dbscan(data: np.ndarray, method: str = "grid_search", **kwargs) -> DBSCAN:
    """
    Optimise les paramètres de DBSCAN en utilisant différentes stratégies.

    Paramètres:
    -----------
    data : array-like
        Les données à analyser
    method : str
        Méthode d'optimisation à utiliser :
        - 'grid_search' : Teste plusieurs combinaisons de paramètres (défaut)
        - 'knee' : Utilise la méthode du coude pour eps
        - 'custom' : Utilise des paramètres personnalisés

    **kwargs : arguments additionnels selon la méthode
        Pour 'grid_search' :
            - eps_range : array-like (défaut=None)
            - min_samples_range : array-like (défaut=None)
            - scoring : str (défaut='silhouette')
        Pour 'knee' :
            - n_neighbors : int (défaut=2)
            - min_samples : int (défaut=5)
        Pour 'custom' :
            - eps : float
            - min_samples : int

    Retourne:
    --------
    DBSCAN : Le meilleur modèle DBSCAN trouvé
    """
    if method == 'grid_search':
        eps_range = kwargs.get('eps_range')
        min_samples_range = kwargs.get('min_samples_range')
        scoring = kwargs.get('scoring', 'silhouette')

        scores = find_optimal_parameters(
            data,
            eps_range=eps_range,
            min_samples_range=min_samples_range,
            method=scoring
        )

        if not scores:
            raise ValueError("Aucune combinaison de paramètres valide trouvée")

        # Trouver les meilleurs paramètres
        best_params = max(scores.items(), key=lambda x: x[1])[0]
        eps, min_samples = best_params

        return DBSCAN(eps=eps, min_samples=min_samples)

    elif method == 'knee':
        from sklearn.neighbors import NearestNeighbors
        n_neighbors = kwargs.get('n_neighbors', 2)
        min_samples = kwargs.get('min_samples', 5)

        # Calculer les distances aux k plus proches voisins
        nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(data)
        distances, _ = nbrs.kneighbors(data)
        distances = distances[:, -1]  # Prendre la distance maximale pour chaque point
        distances = np.sort(distances)

        # Trouver le point de coude
        knee = KneeLocator(
            range(len(distances)),
            distances,
            curve='convex',
            direction='decreasing'
        )
        eps = distances[knee.elbow] if knee.elbow else distances.mean()

        return DBSCAN(eps=eps, min_samples=min_samples)

    elif method == 'custom':
        eps = kwargs.get('eps')
        min_samples = kwargs.get('min_samples')

        if eps is None or min_samples is None:
            raise ValueError("eps et min_samples doivent être fournis pour la méthode 'custom'")

        return DBSCAN(eps=eps, min_samples=min_samples)

    raise ValueError(
        "Méthode non supportée. Utilisez 'grid_search', 'knee' ou 'custom'"
    )


def train_dbscan(data: np.ndarray, optimize: bool = True, save: bool = True, **kwargs) -> tuple[DBSCAN, dict]:
    """
    Fonction principale pour entraîner un modèle DBSCAN.

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
        - eps_range : array-like (défaut=None)
            Plage de valeurs pour eps
        - min_samples_range : array-like (défaut=None)
            Plage de valeurs pour min_samples
        - model_path : str (défaut='models/dbscan_model.pkl')
            Chemin pour sauvegarder le modèle

    Retourne:
    --------
    tuple : (DBSCAN, dict)
        - Le modèle DBSCAN entraîné
        - Dictionnaire contenant les informations sur l'entraînement
    """
    # Configuration par défaut
    default_config = {
        'method': 'grid_search',
        'model_path': 'models/dbscan_model.pkl'
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
            eps_range=kwargs.get('eps_range'),
            min_samples_range=kwargs.get('min_samples_range'),
            method='silhouette'
        )
        plot_parameter_scores(scores, 'silhouette')

    # Création et entraînement du modèle
    model = optimize_dbscan(scaled_data, method=config['method'], **kwargs)
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
        'eps': model.eps,
        'min_samples': model.min_samples,
        'scaler': scaler,
        'silhouette_score': silhouette_score(scaled_data, labels),
        'calinski_harabasz_score': calinski_harabasz_score(scaled_data, labels),
        'davies_bouldin_score': davies_bouldin_score(scaled_data, labels),
        'n_clusters': len(np.unique(labels)) - (1 if -1 in labels else 0),
        'n_noise': np.sum(labels == -1)
    }

    return model, info


def predict_clusters(model: DBSCAN, data: np.ndarray, scaler: StandardScaler | None = None) -> np.ndarray:
    """
    Prédit les clusters pour de nouvelles données.

    Paramètres:
    -----------
    model : DBSCAN
        Le modèle DBSCAN chargé
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


def save_model(model: DBSCAN, path: str = "models/dbscan_model.pkl"):
    """
    Sauvegarde le modèle DBSCAN entraîné.

    Paramètres:
    -----------
    model : DBSCAN
        Le modèle à sauvegarder
    path : str
        Chemin où sauvegarder le modèle
    """
    joblib.dump(model, path)


def load_dbscan(model_path: str, scaler_path: str | None = None) -> tuple[DBSCAN, StandardScaler | None]:
    """
    Charge un modèle DBSCAN pré-entraîné et son scaler associé.

    Paramètres:
    -----------
    model_path : str
        Chemin vers le fichier du modèle DBSCAN sauvegardé
    scaler_path : str, optional
        Chemin vers le fichier du scaler sauvegardé. Si None, cherchera dans le même dossier

    Retourne:
    --------
    tuple : (DBSCAN, StandardScaler)
        - Le modèle DBSCAN chargé
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
