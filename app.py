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
                    # Au lieu de charger à nouveau le fichier, utiliser les données déjà chargées
                    with st.spinner("Préparation des données numériques..."):
                        # Extraire les colonnes numériques du DataFrame déjà chargé
                        import numpy as np
                        import pandas as pd
                        
                        df_to_process = data_to_process.copy()
                        st.write(f"Traitement de {len(df_to_process)} lignes et {len(df_to_process.columns)} colonnes...")
                        
                        # Code similaire à load_and_prepare_data mais adapté pour travailler avec un DataFrame existant
                        numeric_columns = []
                        
                        columns_to_check = sorted(
                            df_to_process.columns,
                            key=lambda x: any(keyword in str(x).lower() for keyword in PRIORITY_KEYWORDS),
                            reverse=True
                        )
                        
                        for col in columns_to_check:
                            try:
                                if pd.api.types.is_numeric_dtype(df_to_process[col]):
                                    numeric_columns.append(col)
                                    continue
                                
                                cleaned_series = (df_to_process[col].astype(str)
                                                .replace(['', 'NA', 'N/A', 'nan', 'NaN', 'None',
                                                        '<NA>', 'undefined', '?'], np.nan)
                                                .str.replace(',', '.')
                                                .str.strip()
                                                .str.replace(r'[^\d.-]', '', regex=True))
                                
                                numeric_series = pd.to_numeric(cleaned_series, errors='coerce')
                                valid_count = numeric_series.notna().sum()
                                valid_ratio = valid_count / len(df_to_process)
                                
                                if valid_ratio > VALID_RATIO_THRESHOLD:
                                    df_to_process[col] = numeric_series
                                    numeric_columns.append(col)
                            except Exception as e:
                                continue
                        
                        if not numeric_columns:
                            raise ValueError("Aucune colonne numérique n'a été trouvée dans le fichier")
                        
                        df_numeric = df_to_process[numeric_columns].copy()
                        
                        for column in df_numeric.columns:
                            df_numeric[column] = df_numeric[column].replace([np.inf, -np.inf], np.nan)
                            median_value = df_numeric[column].median()
                            
                            if pd.isna(median_value):
                                median_value = 0
                            
                            df_numeric[column] = df_numeric[column].fillna(median_value)
                        
                        st.success(f"Préparation terminée: {len(df_numeric.columns)} colonnes numériques extraites")
                    
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
                        params['n_clusters'] = st.slider("Nombre de clusters (k)", min_value=2, max_value=20, value=5, step=1)
                        # Ajout du paramètre n_init pour l'entraînement final
                        params['n_init'] = st.slider("Nombre d'initialisations (n_init)", min_value=1, max_value=20, value=5, step=1)
                        # Ajouter d'autres paramètres K-Means si besoin (init, max_iter)
                        params['k_min'] = 2 # Pour find_optimal_clusters si on l'utilise
                        # S'assurer que k_max soit au moins égal à n_clusters
                        default_k_max = max(10, params['n_clusters'])
                        params['k_max'] = st.number_input("k max pour recherche auto (si k non fixé)", 
                                                         min_value=params['n_clusters'], 
                                                         max_value=30, 
                                                         value=default_k_max)
                        params['optimize_method'] = 'multiple_init' # Utiliser la méthode par défaut de kmeans_training
                        use_optimal_k = st.checkbox("Trouver k optimal automatiquement (ignore le slider k)?", value=False)


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
                                    kmeans_params = {
                                        'n_clusters': params['n_clusters'],
                                        'optimize': use_optimal_k,
                                        'k_min': params['k_min'],
                                        'k_max': params['k_max'],
                                        'n_init': params['n_init'],
                                        'optimize_method': params['optimize_method']
                                    }
                                    all_params['kmeans'] = kmeans_params
                                    
                                    # Appel spécifique à train_kmeans
                                    info = model_comp.train_kmeans(**kmeans_params)
                                    model = model_comp.models['kmeans']
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