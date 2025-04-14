"""
Module pour la préparation des données pour le clustering.
"""
import pandas as pd
from ..data.correlation_analysis import analyze_correlations
from ..features.selection import analyze_and_select_features

def prepare_features_for_clustering(
    df: pd.DataFrame,
    max_categories: int = 30,
    min_unique_ratio: float = 0.05
) -> pd.DataFrame:
    """
    Prépare les features pour la clusterisation en suivant une approche structurée.

    Args:
        df: DataFrame original
        max_categories: Nombre maximum de catégories autorisées pour les variables catégorielles
        min_unique_ratio: Ratio minimum de valeurs uniques pour les variables numériques

    Returns:
        DataFrame préparé pour la clusterisation
    """
    print("1. Prétraitement initial du DataFrame")
    df_clean, preprocess_info = analyze_and_select_features(
        df,
        max_categories=max_categories,
        min_unique_ratio=min_unique_ratio
    )

    print("\n2. Résumé des colonnes conservées:")
    print(f"- Numériques: {len(preprocess_info['feature_types']['numeric'])}")
    print(f"- Catégorielles ordinales: {len(preprocess_info['feature_types']['ordinal'])}")
    print(f"- Catégorielles nominales: {len(preprocess_info['feature_types']['nominal'])}")

    print("\n3. Colonnes numériques optimisées (downcasting):")
    for col, old_type, new_type in preprocess_info['downcasted_columns']:
        print(f"- {col}: {old_type} -> {new_type}")

    print("\n4. Colonnes catégorielles filtrées (> {max_categories} catégories supprimées):")
    categorical_dropped = [col for col, reason in preprocess_info['dropped_columns']
                         if reason == 'too_many_categories']
    for col in categorical_dropped:
        print(f"- {col}")

    ## A REMPLACER PAR LA VERSION COMPLETE
    # print("\n5. Gestion des valeurs manquantes et non numériques...")
    # # Remplacer les valeurs 'unknown' et autres valeurs non numériques par NaN
    # df_clean = df_clean.replace(['unknown', 'UNKNOWN', 'Unknown', 'NULL', 'null', 'None', 'none'], np.nan)

    # # Pour les colonnes numériques, remplacer les NaN par la médiane
    # numeric_cols = preprocess_info['feature_types']['numeric']
    # for col in numeric_cols:
    #     if col in df_clean.columns:
    #         df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    #         df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    # # Pour les colonnes catégorielles, remplacer les NaN par 'missing'
    # categorical_cols = preprocess_info['feature_types']['nominal'] + preprocess_info['feature_types']['ordinal']
    # for col in categorical_cols:
    #     if col in df_clean.columns:
    #         # Si la colonne est catégorielle, ajouter 'missing' aux catégories
    #         if pd.api.types.is_categorical_dtype(df_clean[col]):
    #             df_clean[col] = df_clean[col].cat.add_categories('missing')
    #         df_clean[col] = df_clean[col].fillna('missing')

    print("\n6. Analyse des corrélations entre variables numériques")
    correlations = analyze_correlations(df_clean, plot=True)
    if correlations['strong_correlations']:
        print("\nVariables fortement corrélées à examiner:")
        for corr in correlations['strong_correlations']:
            print(f"- {corr['var1']} - {corr['var2']}: {corr['correlation']:.2f}")

    ## A REMPLACER PAR LA VERSION COMPLETE
    # print("\n7. Encodage des variables catégorielles...")
    # # Encoder les variables catégorielles en valeurs numériques
    # label_encoders = {}
    # for col in categorical_cols:
    #     if col in df_clean.columns:
    #         label_encoders[col] = LabelEncoder()
    #         df_clean[col] = label_encoders[col].fit_transform(df_clean[col].astype(str))

    return df_clean
