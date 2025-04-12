import streamlit as st
import pandas as pd
import os
import numpy as np
from scripts.data.clean_dataset import process_and_analyze_dataset, display_cleaning_summary
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
                    # Utiliser la fonction de clean_dataset.py
                    # Note: process_and_analyze_dataset pourrait nécessiter des ajustements
                    # pour gérer les paramètres via Streamlit ou pour utiliser des valeurs par défaut raisonnables.
                    df_clean, cleaning_info = process_and_analyze_dataset(data_to_process)

                    st.session_state.preprocessed_data = df_clean
                    st.success("Preprocessing terminé !")

                    # Afficher le résumé du nettoyage
                    st.subheader("Résumé du Nettoyage")
                    # Note: display_cleaning_summary print dans la console.
                    # Il faudra peut-être adapter pour afficher dans Streamlit.
                    # Pour l'instant, affichons les infos brutes ou un résumé simple.
                    st.write(f"Dimensions initiales : {cleaning_info.get('data_summary', {}).get('initial_shape', 'N/A')}")
                    st.write(f"Dimensions finales : {cleaning_info.get('final_stats', {}).get('final_shape', df_clean.shape)}")
                    st.write(f"Colonnes supprimées : {len(cleaning_info.get('final_stats', {}).get('columns_removed', []))} Colonnes")
                    st.write(f"Lignes supprimées : {cleaning_info.get('data_summary', {}).get('initial_shape', [0])[0] - df_clean.shape[0]} Lignes")

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
                        params['k_max'] = st.number_input("k max pour recherche auto (si k non fixé)", min_value=params['n_clusters'], max_value=30, value=10)
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
                                # Importer directement KMeans et les métriques
                                from sklearn.cluster import KMeans 
                                from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
                                from scripts.models import kmeans_training, dbscan_training # Garder pour find_optimal_clusters
                                import matplotlib.pyplot as plt
                                from sklearn.decomposition import PCA
                                import time

                                # 1. Standardiser les données sélectionnées
                                scaler = StandardScaler()
                                scaled_data = scaler.fit_transform(data_for_clustering)
                                st.write("Données standardisées pour le clustering.")

                                # 2. Entraîner le modèle choisi
                                model = None
                                info = {}
                                start_time = time.time()

                                if algo == "K-Means":
                                     if use_optimal_k:
                                         # Utiliser find_optimal_clusters avec n_init=1 pour accélérer la recherche
                                         st.info(f"Recherche du k optimal entre {params['k_min']} et {params['k_max']} (n_init=1)...")
                                         scores = kmeans_training.find_optimal_clusters(
                                              scaled_data, 
                                              k_max=params['k_max'], 
                                              k_min=params['k_min'], 
                                              method='silhouette',
                                              n_init=1 # <-- Utilisation de n_init=1 pour la recherche
                                              )
                                         # Afficher le graphique des scores (peut nécessiter adaptation pour Streamlit)
                                         # fig_k, ax_k = plt.subplots()
                                         # ax_k.plot(list(scores.keys()), list(scores.values()), marker='o')
                                         # ax_k.set_xlabel("Nombre de clusters (k)")
                                         # ax_k.set_ylabel("Score Silhouette")
                                         # st.pyplot(fig_k)
                                         if scores:
                                             best_k = max(scores, key=scores.get)
                                             st.success(f"Meilleur k trouvé : {best_k} (Score: {scores[best_k]:.3f})")
                                             params['n_clusters'] = best_k
                                         else:
                                             st.warning("Impossible de déterminer k optimal, utilisation de la valeur du slider.")
                                             # Fallback sur la valeur du slider

                                     # Entraînement final DIRECT avec KMeans et le n_init choisi
                                     st.info(f"Entraînement K-Means avec k={params['n_clusters']} et n_init={params['n_init']}...")
                                     st.write(f"Shape des données pour KMeans : {scaled_data.shape}") # <-- Afficher la shape
                                     model = KMeans(
                                         n_clusters=params['n_clusters'],
                                         n_init=params['n_init'],
                                         random_state=42, # Pour la reproductibilité de l'appel unique
                                         algorithm='elkan' # <-- Essayer l'algorithme Elkan
                                     )
                                     
                                     # Chronométrer l'étape fit
                                     fit_start_time = time.time()
                                     model.fit(scaled_data)
                                     fit_time = time.time() - fit_start_time
                                     st.write(f"Temps pour model.fit() : {fit_time:.2f} s") # <-- Afficher le temps de fit
                                     
                                     labels = model.labels_
                                     
                                     # Calculer les métriques manuellement
                                     info = {
                                         'n_clusters': params['n_clusters'],
                                         'n_init': params['n_init'],
                                         'scaler': scaler, # Garder le scaler si besoin
                                         'silhouette_score': silhouette_score(scaled_data, labels),
                                         'calinski_harabasz_score': calinski_harabasz_score(scaled_data, labels),
                                         'davies_bouldin_score': davies_bouldin_score(scaled_data, labels),
                                         'inertia': model.inertia_
                                     }
                                     # NOTE: L'ancien appel à train_kmeans est supprimé
                                     # model, info = kmeans_training.train_kmeans(...)


                                elif algo == "DBSCAN":
                                     model, info = dbscan_training.train_dbscan(
                                          scaled_data,
                                          optimize=params['optimize'], # False par défaut
                                          save=False, # Ne pas sauvegarder depuis l'UI par défaut
                                          eps=params['eps'],
                                          min_samples=params['min_samples']
                                          # Passer d'autres params si ajoutés à l'UI
                                     )

                                training_time = time.time() - start_time
                                info['training_time'] = training_time # Ajouter le temps d'entraînement aux infos

                                # 3. Afficher les résultats
                                st.subheader("Résultats du Clustering")
                                if model is not None:
                                    labels = model.labels_ if hasattr(model, 'labels_') else None

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