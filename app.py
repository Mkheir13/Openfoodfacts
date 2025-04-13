import streamlit as st
import pandas as pd
import os
import numpy as np
from scripts.data.clean_dataset import process_and_analyze_dataset, display_cleaning_summary
from scripts.features.scaling import load_and_prepare_data, PRIORITY_KEYWORDS, VALID_RATIO_THRESHOLD
from scripts.models.clustering import prepare_features_for_clustering
from scripts.data.sample_large_files import get_reduced_file # Pour suggérer un fichier par défaut

# Configuration de la page Streamlit
st.set_page_config(layout="wide", page_title="Analyse et Clustering OpenFoodFacts")

st.title("Analyse et Clustering de Données OpenFoodFacts")

# Initialiser l'état de session pour stocker les données
if 'preprocessed_data' not in st.session_state:
    st.session_state.preprocessed_data = None
if 'original_file_path' not in st.session_state:
    st.session_state.original_file_path = None
if 'preprocessed_file_path' not in st.session_state:
    st.session_state.preprocessed_file_path = None

# --- Création des onglets ---
tab1, tab2 = st.tabs(["📊 Preprocessing", "🧠 Clustering"])

# --- Onglet Preprocessing ---
with tab1:
    st.header("1. Chargement et Nettoyage des Données")

    # --- Section Chargement ---
    st.subheader("Chargement du fichier CSV")
    uploaded_file = st.file_uploader("Choisissez un fichier CSV", type="csv")

    # Suggestion du fichier échantillon s'il existe
    default_sample_path = "data/reduced/full_datas_100000.csv"
    if os.path.exists(default_sample_path):
        st.info(f"Fichier échantillon trouvé : `{default_sample_path}`. Vous pouvez l'utiliser ou charger un autre fichier.")
        if st.button("Utiliser le fichier échantillon par défaut"):
             st.session_state.original_file_path = default_sample_path
             uploaded_file = None # Reset uploader if default is chosen
             st.success(f"Fichier `{default_sample_path}` sélectionné.")

    # Si un fichier est chargé ou le chemin par défaut est sélectionné
    data_to_process = None
    if uploaded_file is not None:
        try:
            # Sauvegarder temporairement pour que clean_dataset puisse le lire (si nécessaire)
            # ou lire directement si clean_dataset accepte un BytesIO
            # Pour l'instant, on lit en DataFrame
            data_to_process = pd.read_csv(uploaded_file, sep='\t', low_memory=False) # Adapter sep si nécessaire
            st.session_state.original_file_path = uploaded_file.name # Garder trace du nom
            st.success(f"Fichier '{uploaded_file.name}' chargé avec succès.")
        except Exception as e:
            st.error(f"Erreur lors de la lecture du fichier CSV : {e}")
    elif st.session_state.original_file_path and os.path.exists(st.session_state.original_file_path):
         try:
            data_to_process = pd.read_csv(st.session_state.original_file_path, sep='\t', low_memory=False) # Adapter sep si nécessaire
            st.success(f"Fichier '{st.session_state.original_file_path}' chargé.")
         except Exception as e:
            st.error(f"Erreur lors de la lecture du fichier {st.session_state.original_file_path} : {e}")


    if data_to_process is not None:
        st.dataframe(data_to_process.head())

        # --- Section Nettoyage ---
        st.subheader("Nettoyage et Prétraitement")
        if st.button("Lancer le Preprocessing", key="preprocess_button"):
            with st.spinner("Nettoyage en cours... Veuillez patienter."):
                try:
                    # Simplifier pour utiliser directement load_and_prepare_data
                    with st.spinner("Préparation des données numériques..."):
                        # Nous avons besoin de sauvegarder temporairement le DataFrame
                        import tempfile
                        import os
                        
                        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as temp_file:
                            temp_path = temp_file.name
                            # Sauver le DataFrame en CSV
                            data_to_process.to_csv(temp_path, index=False, sep='\t')
                        
                        try:
                            # Maintenant utiliser load_and_prepare_data avec ce fichier temporaire
                            df_numeric = load_and_prepare_data(temp_path)
                            # Supprimer le fichier temporaire
                            os.unlink(temp_path)
                            st.success(f"Préparation terminée: {len(df_numeric.columns)} colonnes numériques extraites")
                        except Exception as e:
                            # Supprimer le fichier temporaire en cas d'erreur
                            os.unlink(temp_path)
                            st.error(f"Erreur lors du preprocessing: {str(e)}")
                            raise e
                    
                    # Étape 2: Utiliser prepare_features_for_clustering pour le traitement avancé
                    # Paramètres configurables
                    max_categories = st.sidebar.slider("Nombre max de catégories", 5, 100, 30)
                    min_unique_ratio = st.sidebar.slider("Ratio min de valeurs uniques", 0.01, 0.30, 0.05)
                    
                    with st.spinner("Préparation des features pour clustering..."):
                        df_clean = prepare_features_for_clustering(
                            df_numeric,
                            max_categories=max_categories,
                            min_unique_ratio=min_unique_ratio
                        )
                        st.success("Préparation des features terminée !")

                    st.session_state.preprocessed_data = df_clean
                    st.success("Preprocessing complet terminé !")

                    # Afficher le résumé du nettoyage
                    st.subheader("Résumé du Nettoyage")
                    # Créer un résumé des opérations
                    cleaning_summary = {
                        "dimensions_initiales": data_to_process.shape,
                        "dimensions_finales": df_clean.shape,
                        "colonnes_supprimées": len(data_to_process.columns) - len(df_clean.columns),
                        "lignes_supprimées": data_to_process.shape[0] - df_clean.shape[0]
                    }
                    
                    st.write(f"Dimensions initiales : {cleaning_summary['dimensions_initiales']}")
                    st.write(f"Dimensions finales : {cleaning_summary['dimensions_finales']}")
                    st.write(f"Colonnes supprimées : {cleaning_summary['colonnes_supprimées']} colonnes")
                    st.write(f"Lignes supprimées : {cleaning_summary['lignes_supprimées']} lignes")

                    # Option de sauvegarde
                    st.subheader("Sauvegarder les données prétraitées")
                    output_dir = "data/preprocessed"
                    os.makedirs(output_dir, exist_ok=True)
                    # Construire un nom de fichier unique basé sur l'original
                    base_name = os.path.splitext(os.path.basename(st.session_state.original_file_path))[0]
                    preprocessed_filename = f"{base_name}_preprocessed.csv"
                    st.session_state.preprocessed_file_path = os.path.join(output_dir, preprocessed_filename)

                    try:
                        df_clean.to_csv(st.session_state.preprocessed_file_path, index=False, sep='\t') # ou ',' selon préférence
                        st.success(f"Données prétraitées sauvegardées dans : `{st.session_state.preprocessed_file_path}`")

                        # Bouton de téléchargement
                        with open(st.session_state.preprocessed_file_path, "rb") as fp:
                             st.download_button(
                                 label="Télécharger le CSV Prétraité",
                                 data=fp,
                                 file_name=preprocessed_filename,
                                 mime="text/csv"
                             )
                    except Exception as e:
                        st.error(f"Erreur lors de la sauvegarde du fichier : {e}")


                except Exception as e:
                    st.error(f"Une erreur est survenue lors du preprocessing : {e}")
                    st.exception(e) # Affiche la trace complète pour le débogage

    # Afficher les données prétraitées si elles existent
    if st.session_state.preprocessed_data is not None:
        st.subheader("Aperçu des données prétraitées")
        st.dataframe(st.session_state.preprocessed_data.head())
        st.info("Les données prétraitées sont prêtes. Vous pouvez passer à l'onglet 'Clustering'.")


# --- Onglet Clustering ---
with tab2:
    st.header("2. Clustering des Données")

    if st.session_state.preprocessed_data is None:
        st.warning("Veuillez d'abord charger et prétraiter des données dans l'onglet 'Preprocessing'.")
        # Option pour charger un fichier pré-traité existant ?
        st.subheader("Ou charger un fichier pré-traité existant")
        uploaded_preprocessed_file = st.file_uploader("Choisissez un fichier CSV pré-traité", type="csv", key="preprocessed_uploader")

        if uploaded_preprocessed_file is not None:
             try:
                 # Lire directement le fichier uploadé
                 st.session_state.preprocessed_data = pd.read_csv(uploaded_preprocessed_file, sep='\t', low_memory=False) # Adapter sep si nécessaire
                 # Garder une trace du nom, pas forcément le chemin complet
                 st.session_state.preprocessed_file_path = uploaded_preprocessed_file.name 
                 st.success(f"Données pré-traitées chargées depuis {uploaded_preprocessed_file.name}")
                 # Rafraîchir la page pour que le reste de l'onglet se mette à jour avec les données chargées
                 st.rerun()
             except Exception as e:
                 st.error(f"Erreur lors du chargement du fichier pré-traité : {e}")


    if st.session_state.preprocessed_data is not None:
        st.success("Données prétraitées disponibles.")
        df_cluster = st.session_state.preprocessed_data

        st.dataframe(df_cluster.head())

        # --- Sélection des colonnes ---
        st.subheader("Sélection des colonnes pour le clustering")
        # Identifier les colonnes numériques potentielles
        numeric_cols = df_cluster.select_dtypes(include=np.number).columns.tolist()
        # Exclure les identifiants ou colonnes non pertinentes si possible (peut nécessiter une heuristique)
        cols_to_exclude_heuristic = ['code', 'id', 'Unnamed: 0'] # A adapter
        available_cols = [col for col in numeric_cols if col not in cols_to_exclude_heuristic]

        if not available_cols:
             st.warning("Aucune colonne numérique appropriée trouvée pour le clustering après le preprocessing.")
        else:
            selected_cols = st.multiselect(
                "Choisissez les colonnes numériques à utiliser :",
                options=available_cols,
                default=available_cols[:min(5, len(available_cols))] # Sélectionner les 5 premières par défaut
            )

            if not selected_cols:
                st.warning("Veuillez sélectionner au moins une colonne pour le clustering.")
            else:
                st.write("Colonnes sélectionnées:", selected_cols)
                data_for_clustering = df_cluster[selected_cols].dropna() # Supprimer les NA restantes sur les colonnes sélectionnées

                if data_for_clustering.empty:
                    st.error("Après suppression des valeurs manquantes sur les colonnes sélectionnées, il ne reste plus de données pour le clustering.")
                else:
                    st.info(f"{len(data_for_clustering)} lignes utilisables pour le clustering après suppression des NA.")

                    # --- Choix de l'algorithme et des hyperparamètres ---
                    st.subheader("Configuration du Clustering")
                    algo = st.selectbox("Choisissez l'algorithme de clustering :", ["K-Means", "DBSCAN"]) # Ajouter GMM, OPTICS plus tard

                    params = {}
                    if algo == "K-Means":
                        # Partie 1: Configuration de l'optimisation du nombre de clusters
                        st.subheader("Optimisation du nombre de clusters")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            params['k_min'] = st.number_input("Nombre minimum de clusters à tester", 
                                                           min_value=2, 
                                                           max_value=10, 
                                                           value=2)
                        with col2:
                            params['k_max'] = st.number_input("Nombre maximum de clusters à tester", 
                                                           min_value=3, 
                                                           max_value=30, 
                                                           value=15)
                        
                        # Méthodes de scoring pour l'optimisation
                        scoring_methods = ['silhouette', 'calinski_harabasz', 'davies_bouldin', 'inertia']
                        params['scoring_method'] = st.selectbox(
                            "Méthode d'évaluation pour le nombre optimal de clusters", 
                            options=scoring_methods,
                            index=0,  # silhouette par défaut
                            help="Silhouette (plus élevé = meilleur), Calinski-Harabasz (plus élevé = meilleur), Davies-Bouldin (plus bas = meilleur), Inertia (plus bas = meilleur)"
                        )
                        
                        params['n_init'] = st.slider("Nombre d'initialisations pour chaque valeur de k", 
                                                 min_value=1, 
                                                 max_value=20, 
                                                 value=10)
                        
                        # Partie 2: Configuration de l'entraînement K-means
                        st.subheader("Configuration de l'entraînement K-means")
                        
                        # Option pour utiliser le nombre optimal ou définir manuellement
                        use_optimal = st.checkbox("Rechercher automatiquement le nombre optimal de clusters", value=True)
                        if not use_optimal:
                            params['n_clusters'] = st.number_input(
                                "Nombre de clusters", 
                                min_value=2, 
                                max_value=50, 
                                value=5
                            )
                        else:
                            params['n_clusters'] = None
                        
                        # Méthodes d'optimisation pour K-means
                        optimize_methods = ['multiple_init', 'grid_search', 'elkan']
                        params['optimize_method'] = st.selectbox(
                            "Méthode d'optimisation des paramètres K-means", 
                            options=optimize_methods,
                            index=0  # multiple_init par défaut
                        )
                        
                        # Paramètres spécifiques selon la méthode d'optimisation
                        if params['optimize_method'] == 'multiple_init':
                            col1, col2 = st.columns(2)
                            with col1:
                                params['max_iter'] = st.slider(
                                    "Nombre maximum d'itérations", 
                                    min_value=100, 
                                    max_value=1000, 
                                    value=300, 
                                    step=100
                                )
                            with col2:
                                params['init'] = st.selectbox(
                                    "Méthode d'initialisation", 
                                    options=['k-means++', 'random'],
                                    index=0
                                )
                        elif params['optimize_method'] == 'grid_search':
                            params['cv'] = st.slider(
                                "Nombre de folds pour validation croisée", 
                                min_value=2, 
                                max_value=10, 
                                value=3
                            )
                        
                        # Paramètres de sauvegarde
                        params['save'] = st.checkbox("Sauvegarder le modèle entraîné", value=True)
                        if params['save']:
                            params['model_path'] = st.text_input(
                                "Chemin pour sauvegarder le modèle", 
                                value="models/kmeans_model.pkl"
                            )
                        
                        # Forcer optimize=True si n_clusters=None
                        params['optimize'] = True if params['n_clusters'] is None else False
                        
                    elif algo == "DBSCAN":
                        params['eps'] = st.slider("Epsilon (eps)", min_value=0.1, max_value=5.0, value=0.5, step=0.1)
                        params['min_samples'] = st.slider("Nombre minimum de points (min_samples)", min_value=2, max_value=50, value=5, step=1)
                        # Ajouter d'autres paramètres DBSCAN (metric, algorithm)
                        params['optimize'] = False # Ne pas optimiser par défaut dans l'UI, utiliser les valeurs fournies


                    # --- Bouton pour lancer le clustering ---
                    if st.button(f"Lancer le Clustering {algo}", key=f"cluster_{algo}"):
                        with st.spinner(f"Clustering {algo} en cours..."):
                            try:
                                # Importer dynamiquement ou avoir les imports en haut
                                from sklearn.preprocessing import StandardScaler
                                # Importer la classe ModelComparison
                                from scripts.models.model_comparison import ModelComparison
                                import matplotlib.pyplot as plt
                                from sklearn.decomposition import PCA
                                import time

                                # 1. Standardiser les données sélectionnées
                                scaler = StandardScaler()
                                scaled_data = scaler.fit_transform(data_for_clustering)
                                st.write("Données standardisées pour le clustering.")

                                # 2. Initialiser et entraîner le modèle avec ModelComparison
                                start_time = time.time()
                                
                                # Créer une instance de ModelComparison
                                model_comp = ModelComparison(data_for_clustering)
                                
                                # Préparation des paramètres pour chaque algorithme
                                all_params = {}
                                
                                if algo == "K-Means":
                                    # Préparation de dictionnaires distincts pour éviter les conflits de paramètres
                                    # 1. Paramètres pour find_optimal_clusters
                                    find_optimal_params = {
                                        'k_min': params['k_min'],
                                        'k_max': params['k_max'],
                                        'method': params['scoring_method'],
                                        'n_init': params['n_init']
                                    }
                                    
                                    # 2. Paramètres pour optimize_kmeans
                                    optimize_params = {
                                        'method': params['optimize_method']
                                    }
                                    
                                    # Ajouter des paramètres spécifiques selon la méthode d'optimisation
                                    if params['optimize_method'] == 'multiple_init':
                                        optimize_params.update({
                                            'max_iter': params.get('max_iter', 300),
                                            'init': params.get('init', 'k-means++')
                                        })
                                    elif params['optimize_method'] == 'grid_search':
                                        optimize_params.update({
                                            'cv': params.get('cv', 3)
                                        })
                                    
                                    # 3. Paramètres généraux pour train_kmeans
                                    kmeans_params = {
                                        'n_clusters': params['n_clusters'],
                                        'optimize': params['optimize'],
                                        'save': params.get('save', True)
                                    }
                                    
                                    # Ajouter le chemin du modèle si sauvegarde activée
                                    if params.get('save', True) and 'model_path' in params:
                                        kmeans_params['model_path'] = params['model_path']
                                    
                                    # Import direct pour éviter les problèmes de paramètres
                                    from scripts.models.kmeans_training import train_kmeans, find_optimal_clusters
                                    
                                    # Si recherche automatique du nombre optimal, afficher le graphique d'analyse
                                    if params['optimize'] and params['n_clusters'] is None:
                                        with st.spinner("Recherche du nombre optimal de clusters..."):
                                            # Calcul des scores pour différents nombres de clusters
                                            scores = find_optimal_clusters(
                                                scaled_data,
                                                **find_optimal_params
                                            )
                                            
                                            # Affichage du graphique d'analyse
                                            st.subheader(f"Analyse du nombre optimal de clusters - {params['scoring_method']}")
                                            fig, ax = plt.subplots(figsize=(10, 6))
                                            ax.plot(list(scores.keys()), list(scores.values()), marker='o')
                                            ax.set_xlabel('Nombre de clusters')
                                            ax.set_ylabel('Score')
                                            ax.set_title(f'Analyse du nombre optimal de clusters - {params["scoring_method"]}')
                                            ax.grid(True)
                                            st.pyplot(fig)
                                            
                                            # Sélection du nombre optimal
                                            optimal_k = max(scores.items(), key=lambda x: x[1])[0]
                                            st.success(f"Nombre optimal de clusters trouvé: {optimal_k}")
                                    
                                    # Entraînement du modèle
                                    with st.spinner("Entraînement K-Means en cours..."):
                                        # Créer une version personnalisée de train_kmeans pour éviter les conflits
                                        def custom_train_kmeans(data, n_clusters=None, optimize=True, save=True,
                                                             k_min=2, k_max=10, scoring_method='silhouette', n_init=10,
                                                             optimize_method='multiple_init', model_path='models/kmeans_model.pkl',
                                                             max_iter=300, init='k-means++', cv=3):
                                            """Version personnalisée de train_kmeans pour éviter les conflits de paramètres"""
                                            from sklearn.preprocessing import StandardScaler
                                            from sklearn.cluster import KMeans
                                            from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
                                            import matplotlib.pyplot as plt
                                            import os
                                            import joblib
                                            import numpy as np
                                            import time
                                            
                                            # Standardisation des données
                                            scaler = StandardScaler()
                                            scaled_data = scaler.fit_transform(data)
                                            
                                            # Si on doit trouver le nombre optimal de clusters
                                            if optimize and n_clusters is None:
                                                scores = {}
                                                
                                                # Évaluer différents nombres de clusters
                                                for k in range(k_min, k_max + 1):
                                                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=n_init)
                                                    labels = kmeans.fit_predict(scaled_data)
                                                    
                                                    # Calculer le score selon la méthode choisie
                                                    if scoring_method == 'silhouette':
                                                        scores[k] = silhouette_score(scaled_data, labels)
                                                    elif scoring_method == 'calinski_harabasz':
                                                        scores[k] = calinski_harabasz_score(scaled_data, labels)
                                                    elif scoring_method == 'davies_bouldin':
                                                        scores[k] = -davies_bouldin_score(scaled_data, labels)
                                                    elif scoring_method == 'inertia':
                                                        scores[k] = -kmeans.inertia_
                                                
                                                # Trouver le meilleur k
                                                n_clusters = max(scores.items(), key=lambda x: x[1])[0]
                                                st.success(f"Nombre optimal de clusters déterminé: {n_clusters}")
                                            
                                            # Entraîner le modèle final avec les meilleurs paramètres
                                            best_model = None
                                            
                                            if optimize_method == 'multiple_init':
                                                best_score = -np.inf
                                                for _ in range(n_init):
                                                    kmeans = KMeans(
                                                        n_clusters=n_clusters,
                                                        init=init,
                                                        max_iter=max_iter,
                                                        random_state=None
                                                    )
                                                    kmeans.fit(scaled_data)
                                                    
                                                    score = silhouette_score(scaled_data, kmeans.labels_)
                                                    if score > best_score:
                                                        best_score = score
                                                        best_model = kmeans
                                            
                                            elif optimize_method == 'grid_search':
                                                from sklearn.model_selection import GridSearchCV
                                                
                                                param_grid = {
                                                    'init': ['k-means++', 'random'],
                                                    'n_init': [10, 20],
                                                    'max_iter': [200, 300, 400],
                                                    'algorithm': ['elkan', 'full']
                                                }
                                                
                                                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                                                grid_search = GridSearchCV(
                                                    kmeans,
                                                    param_grid=param_grid,
                                                    cv=cv,
                                                    scoring='silhouette'
                                                )
                                                grid_search.fit(scaled_data)
                                                best_model = grid_search.best_estimator_
                                            
                                            elif optimize_method == 'elkan':
                                                best_score = -np.inf
                                                for algo in ['elkan', 'full']:
                                                    kmeans = KMeans(
                                                        n_clusters=n_clusters,
                                                        algorithm=algo,
                                                        random_state=42
                                                    )
                                                    kmeans.fit(scaled_data)
                                                    
                                                    score = silhouette_score(scaled_data, kmeans.labels_)
                                                    if score > best_score:
                                                        best_score = score
                                                        best_model = kmeans
                                            
                                            else:
                                                # Méthode par défaut si aucune des méthodes spécifiées n'est choisie
                                                best_model = KMeans(
                                                    n_clusters=n_clusters,
                                                    init=init,
                                                    n_init=n_init,
                                                    random_state=42
                                                ).fit(scaled_data)
                                            
                                            # Sauvegarder le modèle si demandé
                                            if save:
                                                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                                                scaler_path = os.path.join(os.path.dirname(model_path), 'scaler.pkl')
                                                
                                                joblib.dump(best_model, model_path)
                                                joblib.dump(scaler, scaler_path)
                                                st.info(f"Modèle sauvegardé dans {model_path}")
                                            
                                            # Créer le dictionnaire d'informations
                                            labels = best_model.labels_
                                            info = {
                                                'n_clusters': n_clusters,
                                                'optimize_method': optimize_method,
                                                'scaler': scaler,
                                                'silhouette_score': silhouette_score(scaled_data, labels),
                                                'calinski_harabasz_score': calinski_harabasz_score(scaled_data, labels),
                                                'davies_bouldin_score': davies_bouldin_score(scaled_data, labels),
                                                'inertia': best_model.inertia_
                                            }
                                            
                                            return best_model, info
                                        
                                        # Appel de notre fonction personnalisée
                                        # On passe les paramètres avec leurs noms explicites pour éviter tout conflit
                                        model, info_dict = custom_train_kmeans(
                                            data=data_for_clustering,
                                            n_clusters=params['n_clusters'],
                                            optimize=params['optimize'],
                                            save=params.get('save', True),
                                            k_min=find_optimal_params['k_min'],
                                            k_max=find_optimal_params['k_max'],
                                            scoring_method=find_optimal_params['method'],
                                            n_init=find_optimal_params['n_init'],
                                            optimize_method=optimize_params['method'],
                                            model_path=kmeans_params.get('model_path', 'models/kmeans_model.pkl'),
                                            max_iter=optimize_params.get('max_iter', 300),
                                            init=optimize_params.get('init', 'k-means++'),
                                            cv=optimize_params.get('cv', 3)
                                        )
                                        
                                        # Mise à jour des attributs de modèle
                                        model_comp.models['kmeans'] = model
                                        info = info_dict
                                        info['training_time'] = time.time() - start_time
                                        model_comp.results['kmeans'] = info
                                        
                                        labels = model.labels_
                                
                                elif algo == "DBSCAN":
                                    dbscan_params = {
                                        'eps': params['eps'],
                                        'min_samples': params['min_samples'],
                                        'optimize': params['optimize']
                                    }
                                    all_params['dbscan'] = dbscan_params
                                    
                                    # Appel spécifique à train_dbscan
                                    info = model_comp.train_dbscan(**dbscan_params)
                                    model = model_comp.models['dbscan']
                                    labels = model.labels_
                                
                                training_time = time.time() - start_time
                                st.write(f"Temps total d'entraînement : {training_time:.2f} s")

                                # 3. Afficher les résultats
                                st.subheader("Résultats du Clustering")
                                if model is not None:
                                    if labels is not None:
                                        n_clusters_found = len(set(labels)) - (1 if -1 in labels else 0)
                                        n_noise = list(labels).count(-1)

                                        st.metric("Nombre de clusters trouvés", n_clusters_found)
                                        if n_noise > 0:
                                             st.metric("Nombre de points de bruit (DBSCAN)", n_noise)

                                        st.metric("Temps d'entraînement", f"{info.get('training_time', 0):.2f} s")

                                        # Afficher les métriques (si disponibles et calculées dans info)
                                        col1, col2, col3 = st.columns(3)
                                        with col1:
                                            st.metric("Score Silhouette", f"{info.get('silhouette_score', 'N/A'):.3f}")
                                        with col2:
                                            st.metric("Score Calinski-Harabasz", f"{info.get('calinski_harabasz_score', 'N/A'):.0f}")
                                        with col3:
                                            st.metric("Score Davies-Bouldin", f"{info.get('davies_bouldin_score', 'N/A'):.3f}")

                                        # Ajouter les labels au DataFrame original (pour exploration potentielle)
                                        # Attention à l'index si des lignes ont été supprimées par dropna
                                        df_cluster_results = data_for_clustering.copy()
                                        df_cluster_results['cluster'] = labels
                                        st.session_state.clustered_data = df_cluster_results # Sauvegarder pour potentiel usage futur


                                        # 4. Visualisation (PCA 2D)
                                        st.subheader("Visualisation des Clusters (PCA 2D)")
                                        if scaled_data.shape[1] >= 2:
                                            pca = PCA(n_components=2, random_state=42)
                                            data_2d = pca.fit_transform(scaled_data)

                                            fig, ax = plt.subplots(figsize=(10, 8))

                                            # Scatter plot avec couleurs par cluster
                                            scatter = ax.scatter(data_2d[:, 0], data_2d[:, 1], c=labels, cmap='viridis', alpha=0.6, s=10)

                                            # Distinguer les points de bruit (si DBSCAN)
                                            if n_noise > 0:
                                                 noise_points = data_2d[labels == -1]
                                                 ax.scatter(noise_points[:, 0], noise_points[:, 1], color='red', marker='x', label='Bruit', s=20)

                                            ax.set_title(f"Clusters {algo} visualisés avec PCA")
                                            ax.set_xlabel("Composante Principale 1")
                                            ax.set_ylabel("Composante Principale 2")
                                            # Ajouter une légende si possible/pertinent
                                            # legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
                                            # ax.add_artist(legend1)
                                            if n_noise > 0:
                                                ax.legend()

                                            st.pyplot(fig)
                                        else:
                                            st.warning("La visualisation PCA nécessite au moins 2 colonnes sélectionnées.")
                                        
                                        # Ajout d'une visualisation 3D avec PCA
                                        st.subheader("Visualisation des Clusters en 3D (PCA)")
                                        if scaled_data.shape[1] >= 3:
                                            from mpl_toolkits.mplot3d import Axes3D
                                            
                                            # Réduire à 3 dimensions avec PCA
                                            pca3d = PCA(n_components=3, random_state=42)
                                            data_3d = pca3d.fit_transform(scaled_data)
                                            
                                            # Créer une figure 3D
                                            fig3d = plt.figure(figsize=(10, 8))
                                            ax3d = fig3d.add_subplot(111, projection='3d')
                                            
                                            # Scatter plot 3D avec couleurs par cluster
                                            scatter3d = ax3d.scatter(
                                                data_3d[:, 0], 
                                                data_3d[:, 1], 
                                                data_3d[:, 2],
                                                c=labels, 
                                                cmap='viridis', 
                                                alpha=0.6,
                                                s=10
                                            )
                                            
                                            # Distinguer les points de bruit (si DBSCAN)
                                            if n_noise > 0:
                                                noise_points_3d = data_3d[labels == -1]
                                                ax3d.scatter(
                                                    noise_points_3d[:, 0],
                                                    noise_points_3d[:, 1],
                                                    noise_points_3d[:, 2],
                                                    color='red',
                                                    marker='x',
                                                    label='Bruit',
                                                    s=20
                                                )
                                                ax3d.legend()
                                            
                                            ax3d.set_title(f"Clusters {algo} visualisés en 3D avec PCA")
                                            ax3d.set_xlabel("Composante Principale 1")
                                            ax3d.set_ylabel("Composante Principale 2")
                                            ax3d.set_zlabel("Composante Principale 3")
                                            
                                            # Ajouter une barre de couleur
                                            cbar = fig3d.colorbar(scatter3d, ax=ax3d, pad=0.1)
                                            cbar.set_label('Clusters')
                                            
                                            st.pyplot(fig3d)
                                        elif scaled_data.shape[1] == 2:
                                            st.warning("La visualisation 3D nécessite au moins 3 colonnes sélectionnées. Actuellement vous n'avez que 2 colonnes.")
                                        else:
                                            st.warning("La visualisation 3D nécessite au moins 3 colonnes sélectionnées.")

                                        # Afficher un aperçu des données avec les clusters assignés
                                        st.subheader("Aperçu des données avec clusters")
                                        st.dataframe(df_cluster_results.head())

                                    else:
                                        st.error("Le modèle n'a pas retourné de labels.")
                                else:
                                     st.error("L'entraînement du modèle a échoué.")

                            except Exception as e:
                                st.error(f"Une erreur est survenue lors du clustering : {e}")
                                st.exception(e) # Afficher la trace pour débogage 