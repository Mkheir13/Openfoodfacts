import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.data.preprocessing import handle_duplicates, process_categorical_columns
from scripts.models.kmeans_training import find_optimal_clusters, optimize_kmeans, train_kmeans
from scripts.models.dbscan_training import train_dbscan
from scripts.models.gmm_training import train_gmm
from scripts.models.optics_training import train_optics
from scripts.models.clustering import prepare_features_for_clustering

# Fonction pour échantillonner un fichier CSV volumineux
def sample_large_csv(file_path, sample_size=10000, random_state=42):
    """
    Échantillonne un fichier CSV volumineux en lisant seulement une partie des lignes.
    
    Args:
        file_path: Chemin vers le fichier CSV
        sample_size: Nombre de lignes à échantillonner
        random_state: État aléatoire pour la reproductibilité
        
    Returns:
        DataFrame échantillonné
    """
    # Compter le nombre total de lignes (plus rapide que pd.read_csv sur tout le fichier)
    total_rows = sum(1 for _ in open(file_path, 'r', encoding='utf-8'))
    
    # Calculer le taux d'échantillonnage
    if total_rows <= sample_size:
        return pd.read_csv(file_path, low_memory=False)
    
    # Calculer la fraction pour l'échantillonnage
    fraction = sample_size / total_rows
    
    # Échantillonner le fichier
    return pd.read_csv(file_path, skiprows=lambda x: x > 0 and np.random.random() > fraction, low_memory=False)

# Set page config
st.set_page_config(
    page_title="Openfoodfacts Cluster Visualization",
    page_icon="🍎",
    layout="wide"
)

# Title and introduction
st.title("Openfoodfacts Cluster Visualization")
st.markdown("""
Cette application permet de visualiser les clusters des produits Openfoodfacts avec des paramètres ajustables.
Vous pouvez sélectionner différents algorithmes de clustering et ajuster les paramètres pour obtenir des résultats optimaux.
""")

# Data loading section
st.header("1. Chargement des données")

# Option to upload a file or use an example dataset
data_option = st.radio(
    "Sélectionner la source de données:",
    ["Charger un fichier", "Utiliser un exemple (si disponible)"]
)

df = None

if data_option == "Charger un fichier":
    uploaded_file = st.file_uploader("Charger un fichier CSV ou Excel", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        # Ajouter un contrôle pour la taille d'échantillon
        col1, col2 = st.columns(2)
        with col1:
            file_size_mb = uploaded_file.size / (1024 * 1024)
            st.info(f"Taille du fichier: {file_size_mb:.2f} MB")
            
            if file_size_mb > 100:
                st.warning("Le fichier est volumineux, un échantillonnage est recommandé.")
        
        with col2:
            sample_data = st.checkbox(
                "Échantillonner les données (pour fichiers volumineux)", 
                value=file_size_mb > 100 if 'file_size_mb' in locals() else False
            )
            
            if sample_data:
                sample_size = st.number_input(
                    "Nombre de lignes à échantillonner",
                    min_value=1000,
                    max_value=100000,
                    value=10000,
                    step=1000
                )
        
        try:
            # Traitement selon le type de fichier
            if uploaded_file.name.endswith('.csv'):
                if sample_data:
                    # Sauvegarder temporairement le fichier pour l'échantillonnage
                    temp_file_path = "temp_upload.csv"
                    with open(temp_file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Échantillonner le fichier
                    df = sample_large_csv(temp_file_path, sample_size=sample_size)
                    
                    # Supprimer le fichier temporaire
                    if os.path.exists(temp_file_path):
                        os.remove(temp_file_path)
                else:
                    df = pd.read_csv(uploaded_file, low_memory=False)
            else:  # Excel
                if sample_data:
                    # Pour Excel, utilisons directement pandas avec nrows
                    df = pd.read_excel(uploaded_file, nrows=sample_size)
                else:
                    df = pd.read_excel(uploaded_file)
            
            st.success(f"Fichier chargé avec succès: {uploaded_file.name}")
            if sample_data:
                st.info(f"Échantillon de {len(df)} lignes sur le fichier original")
        except Exception as e:
            st.error(f"Erreur lors du chargement du fichier: {e}")
else:
    # Try to load example data from the data directory
    reduced_dir = "data/reduced"
    data_dir = "data"
    
    # Rechercher d'abord des versions réduites prétraitées
    example_paths = []
    
    # Versions réduites (préférées)
    if os.path.exists(reduced_dir):
        reduced_files = [f for f in os.listdir(reduced_dir) if f.endswith('.csv')]
        for rf in reduced_files:
            example_paths.append(os.path.join(reduced_dir, rf))
    
    # Fichiers standards
    example_paths.extend([
        os.path.join(data_dir, "example_data.csv"),
        os.path.join(data_dir, "openfoodfacts.csv"),
        os.path.join(data_dir, "openfoodfacts_100000_10000.csv")  # Version réduite
    ])
    
    # Proposer un sélecteur si plusieurs exemples sont disponibles
    available_examples = [path for path in example_paths if os.path.exists(path)]
    
    if available_examples:
        if len(available_examples) > 1:
            # Afficher les noms des fichiers sans le chemin complet
            example_names = [os.path.basename(path) for path in available_examples]
            selected_example = st.selectbox("Sélectionner un exemple:", example_names)
            
            # Retrouver le chemin complet
            selected_path = next(path for path, name in zip(available_examples, example_names) 
                              if name == selected_example)
        else:
            selected_path = available_examples[0]
            st.info(f"Exemple disponible : {os.path.basename(selected_path)}")
        
        try:
            # Vérifier la taille du fichier
            file_size_mb = os.path.getsize(selected_path) / (1024 * 1024)
            
            if file_size_mb > 100:
                # Pour les fichiers volumineux, proposer l'échantillonnage
                st.warning(f"Le fichier est volumineux ({file_size_mb:.1f} MB), un échantillonnage est recommandé.")
                sample_size = st.number_input(
                    "Nombre de lignes à échantillonner",
                    min_value=1000,
                    max_value=100000,
                    value=10000,
                    step=1000
                )
                
                # Importer uniquement le nombre spécifié de lignes
                df = pd.read_csv(selected_path, nrows=sample_size, low_memory=False)
                st.success(f"Échantillon de {len(df)} lignes chargé depuis {os.path.basename(selected_path)}")
            else:
                # Pour les petits fichiers, charger tout
                df = pd.read_csv(selected_path, low_memory=False)
                st.success(f"Exemple chargé : {os.path.basename(selected_path)}")
                
        except Exception as e:
            st.error(f"Erreur lors du chargement de l'exemple : {e}")
    else:
        st.warning("Aucun exemple disponible. Veuillez charger un fichier.")

# Display data info if loaded
if df is not None:
    with st.expander("Aperçu des données"):
        st.write(f"Dimensions: {df.shape[0]} lignes × {df.shape[1]} colonnes")
        st.dataframe(df.head())
        
        st.subheader("Types de données")
        st.dataframe(pd.DataFrame({
            'Type': df.dtypes,
            'Non-Null Count': df.count(),
            'Null Count': df.isnull().sum(),
            'Unique Values': df.nunique()
        }))

    # Data preprocessing section
    st.header("2. Prétraitement des données")
    
    with st.form("preprocessing_form"):
        st.subheader("Options de prétraitement")
        
        col1, col2 = st.columns(2)
        
        with col1:
            duplicate_strategy = st.selectbox(
                "Stratégie pour les doublons:",
                ["analyze", "remove_all", "keep_first", "keep_last", "aggregate"],
                index=2
            )
            
            max_categories = st.slider(
                "Nombre maximum de catégories pour variables catégorielles:",
                min_value=5,
                max_value=100,
                value=30,
                step=5
            )
        
        with col2:
            min_unique_ratio = st.slider(
                "Ratio minimum de valeurs uniques pour variables numériques:",
                min_value=0.01,
                max_value=0.5,
                value=0.05,
                step=0.01
            )
            
            handle_missing = st.checkbox("Gérer les valeurs manquantes", value=True)
            
            if handle_missing:
                missing_strategy = st.selectbox(
                    "Méthode de traitement des valeurs manquantes:",
                    ["Imputation automatique", "Suppression des lignes", "Détection automatique"],
                    index=2,
                    help="Imputation: remplace les valeurs manquantes par la médiane/mode. Suppression: élimine les lignes avec des NA. Détection auto: choisit la méthode selon les données."
                )
                
                filter_na_cols = st.checkbox("Filtrer les colonnes avec trop de valeurs manquantes", value=True)
                if filter_na_cols:
                    max_na_pct = st.slider(
                        "% maximum de valeurs manquantes par colonne:",
                        min_value=10,
                        max_value=95,
                        value=80,
                        step=5,
                        help="Les colonnes ayant plus que ce pourcentage de valeurs manquantes seront supprimées"
                    )
        
        preprocess_button = st.form_submit_button("Prétraiter les données")
    
    if preprocess_button:
        with st.spinner("Prétraitement en cours..."):
            # Handle duplicates
            df_clean, duplicate_info = handle_duplicates(df, strategy=duplicate_strategy)
            
            # Process categorical columns
            df_clean, cat_info = process_categorical_columns(df_clean, max_categories=max_categories)
            
            # Handle missing values if requested
            if handle_missing:
                # Obtenir des statistiques sur les valeurs manquantes
                missing_before = df_clean.isnull().sum().sum()
                rows_with_na = df_clean.isnull().any(axis=1).sum()
                pct_rows_with_na = rows_with_na / len(df_clean) if len(df_clean) > 0 else 0
                
                # Filtrer les colonnes avec trop de valeurs manquantes si demandé
                if filter_na_cols:
                    # Calculer le pourcentage de valeurs manquantes par colonne
                    na_pct = df_clean.isnull().mean() * 100
                    cols_to_drop = na_pct[na_pct > max_na_pct].index.tolist()
                    
                    if cols_to_drop:
                        st.warning(f"Suppression de {len(cols_to_drop)} colonnes avec plus de {max_na_pct}% de valeurs manquantes")
                        for col in cols_to_drop:
                            st.text(f"  - {col}: {na_pct[col]:.1f}% manquantes")
                        
                        df_clean = df_clean.drop(columns=cols_to_drop)
                        
                        # Recalculer les statistiques après filtrage
                        missing_after_filter = df_clean.isnull().sum().sum()
                        st.info(f"Valeurs manquantes après filtrage des colonnes: {missing_after_filter} (réduction de {missing_before - missing_after_filter} valeurs)")
                        missing_before = missing_after_filter
                        rows_with_na = df_clean.isnull().any(axis=1).sum()
                        pct_rows_with_na = rows_with_na / len(df_clean) if len(df_clean) > 0 else 0
                
                # Choisir la stratégie à appliquer
                if missing_strategy == "Détection automatique":
                    # Si plus de 80% des lignes ont des valeurs manquantes, imputation
                    if pct_rows_with_na > 0.8:
                        actual_strategy = "Imputation automatique"
                    else:
                        actual_strategy = "Suppression des lignes"
                    st.info(f"Stratégie automatiquement sélectionnée : {actual_strategy} ({pct_rows_with_na:.1%} de lignes avec NA)")
                else:
                    actual_strategy = missing_strategy
                
                # Appliquer la stratégie choisie
                if actual_strategy == "Imputation automatique":
                    # Remplir les valeurs manquantes par la médiane pour les colonnes numériques
                    numeric_cols = df_clean.select_dtypes(include=['number']).columns
                    for col in numeric_cols:
                        df_clean[col] = df_clean[col].fillna(df_clean[col].median() if not df_clean[col].isnull().all() else 0)
                    
                    # Remplir les valeurs manquantes par le mode pour les colonnes catégorielles
                    cat_cols = df_clean.select_dtypes(exclude=['number']).columns
                    for col in cat_cols:
                        df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0] if not df_clean[col].mode().empty else "INCONNU")
                    
                    missing_after = df_clean.isnull().sum().sum()
                    st.info(f"Valeurs manquantes: {missing_before} avant, {missing_after} après imputation")
                else:
                    # Suppression des lignes avec NA
                    df_clean = df_clean.dropna()
                    missing_after = df_clean.isnull().sum().sum()
                    missing_dropped = missing_before - missing_after
                    st.info(f"Valeurs manquantes: {missing_before} avant, {missing_dropped} supprimées")
                
                # Vérifier qu'il reste des données après traitement
                if df_clean.empty:
                    st.error("Toutes les lignes ont été supprimées lors du traitement des valeurs manquantes. Essayez d'utiliser l'imputation au lieu de la suppression.")
                    st.stop()
            
            # Prepare features for clustering
            df_cluster = prepare_features_for_clustering(
                df_clean, 
                max_categories=max_categories,
                min_unique_ratio=min_unique_ratio
            )
            
            # Vérifier que le DataFrame n'est pas vide
            if df_cluster.empty:
                st.error("Après prétraitement, aucune donnée n'est disponible pour le clustering. Veuillez ajuster les paramètres de prétraitement.")
                st.stop()
            
            st.success("Prétraitement terminé!")
            st.session_state['df_cluster'] = df_cluster
            st.session_state['df_clean'] = df_clean
    
    # Only show clustering section if preprocessed data is available
    if 'df_cluster' in st.session_state:
        df_cluster = st.session_state['df_cluster']
        
        # Clustering section
        st.header("3. Configuration du clustering")
        
        # Select algorithm
        algorithm = st.selectbox(
            "Algorithme de clustering:",
            ["K-Means", "DBSCAN", "GMM (Gaussian Mixture Models)", "OPTICS"],
            index=0
        )
        
        # Algorithm-specific parameters
        if algorithm == "K-Means":
            with st.expander("Paramètres K-Means", expanded=True):
                find_optimal = st.checkbox("Trouver le nombre optimal de clusters", value=True)
                
                if find_optimal:
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        k_min = st.number_input("Nombre minimum de clusters", min_value=2, value=2)
                    
                    with col2:
                        k_max = st.number_input("Nombre maximum de clusters", min_value=3, value=10)
                    
                    with col3:
                        method = st.selectbox(
                            "Méthode d'évaluation:",
                            ["silhouette", "calinski_harabasz", "davies_bouldin", "inertia"],
                            index=0
                        )
                else:
                    n_clusters = st.slider("Nombre de clusters", min_value=2, max_value=20, value=5)
                
                optimization_method = st.selectbox(
                    "Méthode d'optimisation:",
                    ["multiple_init", "grid_search", "elkan"],
                    index=0
                )
        
        elif algorithm == "DBSCAN":
            with st.expander("Paramètres DBSCAN", expanded=True):
                col1, col2 = st.columns(2)
                
                with col1:
                    eps = st.slider("Epsilon (distance max entre points)", 
                                    min_value=0.1, max_value=5.0, value=0.5, step=0.1)
                
                with col2:
                    min_samples = st.slider("Minimum de points dans un cluster", 
                                            min_value=2, max_value=20, value=5)
        
        elif algorithm == "GMM (Gaussian Mixture Models)":
            with st.expander("Paramètres GMM", expanded=True):
                col1, col2 = st.columns(2)
                
                with col1:
                    n_components = st.slider("Nombre de composantes (clusters)", 
                                            min_value=2, max_value=20, value=5)
                
                with col2:
                    covariance_type = st.selectbox(
                        "Type de covariance:",
                        ["full", "tied", "diag", "spherical"],
                        index=0
                    )
        
        elif algorithm == "OPTICS":
            with st.expander("Paramètres OPTICS", expanded=True):
                col1, col2 = st.columns(2)
                
                with col1:
                    min_samples_optics = st.slider("Minimum de points dans un cluster", 
                                                  min_value=2, max_value=20, value=5)
                
                with col2:
                    max_eps_optics = st.slider("Epsilon maximum", 
                                               min_value=0.5, max_value=10.0, value=2.0, step=0.1)
        
        # Feature selection for clustering
        st.subheader("Sélection des caractéristiques")
        
        feature_cols = df_cluster.columns.tolist()
        num_features = df_cluster.select_dtypes(include=['number']).columns.tolist()
        
        selected_features = st.multiselect(
            "Sélectionner les caractéristiques pour le clustering",
            options=feature_cols,
            default=num_features[:min(10, len(num_features))]
        )
        
        if not selected_features:
            st.warning("Veuillez sélectionner au moins deux caractéristiques.")
        
        # Dimensionality reduction for visualization
        st.subheader("Réduction de dimensionnalité pour visualisation")
        
        dim_reduction = st.selectbox(
            "Méthode de réduction:",
            ["PCA", "Sélection manuelle de 2 dimensions"],
            index=0
        )
        
        if dim_reduction == "Sélection manuelle de 2 dimensions" and len(selected_features) >= 2:
            col1, col2 = st.columns(2)
            
            with col1:
                x_feature = st.selectbox("Dimension X:", options=selected_features, index=0)
            
            with col2:
                remaining_features = [f for f in selected_features if f != x_feature]
                y_feature = st.selectbox("Dimension Y:", options=remaining_features, 
                                         index=0 if remaining_features else None)
        
        # Run clustering button
        if len(selected_features) >= 2:
            run_clustering = st.button("Exécuter le clustering")
            
            if run_clustering:
                with st.spinner("Clustering en cours..."):
                    # Prepare data for clustering
                    X = df_cluster[selected_features].values
                    
                    # Vérifier que X n'est pas vide
                    if X.shape[0] == 0:
                        st.error("Aucune donnée disponible pour le clustering. Veuillez vérifier vos filtres ou critères de sélection.")
                        st.stop()
                    
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    
                    # Store in session state
                    st.session_state['X_scaled'] = X_scaled
                    st.session_state['selected_features'] = selected_features
                    
                    # Run clustering algorithm
                    if algorithm == "K-Means":
                        if find_optimal:
                            with st.spinner("Recherche du nombre optimal de clusters..."):
                                scores = find_optimal_clusters(
                                    X_scaled, k_max=k_max, k_min=k_min, method=method
                                )
                                
                                # Plot scores
                                fig, ax = plt.subplots(figsize=(10, 6))
                                ax.plot(list(scores.keys()), list(scores.values()), marker='o')
                                ax.set_xlabel('Nombre de clusters')
                                ax.set_ylabel('Score')
                                ax.set_title(f'Analyse du nombre optimal de clusters - {method}')
                                ax.grid(True)
                                
                                st.pyplot(fig)
                                
                                # Find optimal k
                                if method in ['silhouette', 'calinski_harabasz']:
                                    optimal_k = max(scores.items(), key=lambda x: x[1])[0]
                                else:  # davies_bouldin, inertia
                                    optimal_k = min(scores.items(), key=lambda x: x[1])[0]
                                
                                st.info(f"Nombre optimal de clusters: {optimal_k}")
                                n_clusters = optimal_k
                        
                        # Train model with optimized parameters
                        model = optimize_kmeans(
                            X_scaled, n_clusters=n_clusters, method=optimization_method
                        )
                        labels = model.labels_
                    
                    elif algorithm == "DBSCAN":
                        model = train_dbscan(
                            X_scaled, eps=eps, min_samples=min_samples, return_model=True
                        )
                        labels = model.labels_
                    
                    elif algorithm == "GMM (Gaussian Mixture Models)":
                        model = train_gmm(
                            X_scaled, n_components=n_components, 
                            covariance_type=covariance_type, return_model=True
                        )
                        labels = model.predict(X_scaled)
                    
                    elif algorithm == "OPTICS":
                        model = train_optics(
                            X_scaled, min_samples=min_samples_optics, 
                            max_eps=max_eps_optics, return_model=True
                        )
                        labels = model.labels_
                    
                    # Store results in session state
                    st.session_state['labels'] = labels
                    st.session_state['model'] = model
                    
                    # Process visualization
                    if dim_reduction == "PCA":
                        pca = PCA(n_components=2)
                        X_2d = pca.fit_transform(X_scaled)
                        explained_variance = pca.explained_variance_ratio_
                        
                        st.session_state['X_2d'] = X_2d
                        st.session_state['explained_variance'] = explained_variance
                        st.session_state['viz_method'] = 'PCA'
                    else:
                        x_idx = selected_features.index(x_feature)
                        y_idx = selected_features.index(y_feature)
                        X_2d = X_scaled[:, [x_idx, y_idx]]
                        
                        st.session_state['X_2d'] = X_2d
                        st.session_state['x_feature'] = x_feature
                        st.session_state['y_feature'] = y_feature
                        st.session_state['viz_method'] = 'manual'
                    
                    st.success("Clustering terminé! Consultez les résultats ci-dessous.")
        
        # Results section - only show if clustering has been run
        if 'labels' in st.session_state:
            st.header("4. Résultats du clustering")
            
            # Get data from session state
            labels = st.session_state['labels']
            X_2d = st.session_state['X_2d']
            
            # Count clusters
            unique_labels = np.unique(labels)
            n_clusters_found = len(unique_labels)
            if -1 in unique_labels:  # DBSCAN/OPTICS noise points
                n_clusters_found -= 1
            
            st.write(f"Nombre de clusters trouvés: {n_clusters_found}")
            
            if -1 in unique_labels:
                noise_count = np.sum(labels == -1)
                st.write(f"Points de bruit (non classifiés): {noise_count} ({noise_count/len(labels):.2%})")
            
            # Visualization
            st.subheader("Visualisation des clusters")
            
            viz_options = ["Scatter Plot", "3D Scatter (si disponible)", "Distribution par cluster"]
            viz_type = st.selectbox("Type de visualisation:", viz_options)
            
            if viz_type == "Scatter Plot":
                # Create DataFrame for plotting
                viz_df = pd.DataFrame({
                    'x': X_2d[:, 0],
                    'y': X_2d[:, 1],
                    'cluster': labels
                })
                
                # Add axis labels based on viz method
                if st.session_state['viz_method'] == 'PCA':
                    explained_variance = st.session_state['explained_variance']
                    x_label = f"PC1 ({explained_variance[0]:.2%})"
                    y_label = f"PC2 ({explained_variance[1]:.2%})"
                else:
                    x_label = st.session_state['x_feature']
                    y_label = st.session_state['y_feature']
                
                # Create Plotly figure
                fig = px.scatter(
                    viz_df, x='x', y='y', color='cluster',
                    labels={'x': x_label, 'y': y_label, 'cluster': 'Cluster'},
                    title="Visualisation des clusters",
                    color_continuous_scale=px.colors.qualitative.G10 
                    if -1 not in unique_labels else px.colors.qualitative.Bold
                )
                
                # Improve layout
                fig.update_layout(
                    plot_bgcolor='white',
                    height=600,
                    legend=dict(
                        title="Cluster",
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            elif viz_type == "Distribution par cluster":
                if len(st.session_state['selected_features']) >= 3:
                    feature_for_dist = st.selectbox(
                        "Caractéristique pour la distribution:",
                        options=st.session_state['selected_features']
                    )
                    
                    # Get the feature values
                    feat_idx = st.session_state['selected_features'].index(feature_for_dist)
                    feat_values = st.session_state['X_scaled'][:, feat_idx]
                    
                    # Create distribution plot
                    fig = go.Figure()
                    
                    for cluster in unique_labels:
                        cluster_name = "Bruit" if cluster == -1 else f"Cluster {cluster}"
                        mask = labels == cluster
                        
                        fig.add_trace(go.Histogram(
                            x=feat_values[mask],
                            name=cluster_name,
                            opacity=0.7,
                            nbinsx=30
                        ))
                    
                    fig.update_layout(
                        title=f"Distribution de {feature_for_dist} par cluster",
                        xaxis_title=feature_for_dist,
                        yaxis_title="Nombre d'observations",
                        barmode='overlay',
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Cette visualisation nécessite au moins 3 caractéristiques.")
            
            elif viz_type == "3D Scatter (si disponible)":
                if len(st.session_state['selected_features']) >= 3:
                    # If we have at least 3 features, create a 3D scatter
                    if st.session_state['viz_method'] == 'PCA':
                        # Need to recalculate PCA with 3 components
                        X_scaled = st.session_state['X_scaled']
                        pca = PCA(n_components=3)
                        X_3d = pca.fit_transform(X_scaled)
                        explained_variance = pca.explained_variance_ratio_
                        
                        x_label = f"PC1 ({explained_variance[0]:.2%})"
                        y_label = f"PC2 ({explained_variance[1]:.2%})"
                        z_label = f"PC3 ({explained_variance[2]:.2%})"
                    else:
                        # Select a third feature
                        selected_features = st.session_state['selected_features']
                        x_feature = st.session_state['x_feature']
                        y_feature = st.session_state['y_feature']
                        remaining_features = [f for f in selected_features 
                                             if f not in [x_feature, y_feature]]
                        
                        z_feature = st.selectbox(
                            "Sélectionnez la 3ème dimension:",
                            options=remaining_features,
                            index=0 if remaining_features else None
                        )
                        
                        if z_feature:
                            X_scaled = st.session_state['X_scaled']
                            x_idx = selected_features.index(x_feature)
                            y_idx = selected_features.index(y_feature)
                            z_idx = selected_features.index(z_feature)
                            
                            X_3d = np.column_stack([
                                X_scaled[:, x_idx], 
                                X_scaled[:, y_idx],
                                X_scaled[:, z_idx]
                            ])
                            
                            x_label = x_feature
                            y_label = y_feature
                            z_label = z_feature
                        else:
                            st.warning("Veuillez sélectionner une 3ème dimension.")
                            z_feature = None
                    
                    # Create 3D scatter plot
                    if z_feature is not None:
                        viz_df = pd.DataFrame({
                            'x': X_3d[:, 0],
                            'y': X_3d[:, 1],
                            'z': X_3d[:, 2],
                            'cluster': labels
                        })
                        
                        fig = px.scatter_3d(
                            viz_df, x='x', y='y', z='z', color='cluster',
                            labels={'x': x_label, 'y': y_label, 'z': z_label, 'cluster': 'Cluster'},
                            title="Visualisation 3D des clusters",
                            opacity=0.7
                        )
                        
                        # Improve layout
                        fig.update_layout(height=700)
                        
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Cette visualisation nécessite au moins 3 caractéristiques.")
    else:
        st.info("Veuillez prétraiter les données pour passer au clustering.")
else:
    st.info("Veuillez charger un jeu de données pour commencer.")

# Add footer
st.markdown("---")
st.markdown("**Openfoodfacts Cluster Visualization** - Application créée avec Streamlit") 