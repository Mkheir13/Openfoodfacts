"""
Module pour le sous-échantillonnage intelligent et stratifié des datasets.

Ce module fournit des fonctions pour analyser et sous-échantillonner des datasets
de manière intelligente en utilisant différentes stratégies, notamment le
sous-échantillonnage stratifié.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def _auto_select_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """
    Sélectionne intelligemment les meilleures colonnes pour la stratification
    et l'analyse numérique.

    La sélection est basée sur plusieurs critères :
    1. Pour les colonnes catégorielles :
        - Nombre de catégories uniques (entre 2 et 50)
        - Ratio de valeurs non nulles (> 30%)
        - Distribution équilibrée des catégories
    2. Pour les colonnes numériques :
        - Ratio de valeurs non nulles (> 30%)
        - Ratio de valeurs uniques (entre 1% et 90%)
        - Variance significative
        - Absence de valeurs extrêmes excessives

    Args:
        df: DataFrame à analyser

    Returns:
        Tuple[List[str], List[str]]: (colonnes de stratification, colonnes numériques)
    """
    # Colonnes à exclure automatiquement (par pattern)
    exclude_patterns = [
        '_t$', '_datetime$', 'url', 'code', 'creator', 'created',
        'modified', 'updated', 'id', 'uuid', 'guid', 'hash', 'key',
        'index', 'timestamp', 'date', 'time', 'version'
    ]

    def is_excluded(col: str) -> bool:
        """Vérifie si une colonne doit être exclue."""
        col_lower = col.lower()
        return any(pattern in col_lower for pattern in exclude_patterns)

    def calculate_distribution_score(series: pd.Series) -> float:
        """Calcule un score d'équilibre de distribution pour une série."""
        counts = series.value_counts(normalize=True)
        entropy = -(counts * np.log(counts)).sum()  # Entropie de Shannon
        max_entropy = np.log(len(counts))  # Entropie maximale possible
        return entropy / max_entropy if max_entropy > 0 else 0

    def calculate_numeric_score(series: pd.Series) -> float:
        """Calcule un score de qualité pour une colonne numérique."""
        if series.std() == 0:  # Évite les colonnes constantes
            return 0

        # Calcul des z-scores pour détecter les valeurs extrêmes
        z_scores = np.abs((series - series.mean()) / series.std())
        outlier_ratio = (z_scores > 3).mean()

        # Normalisation des critères entre 0 et 1
        non_null_score = series.notna().mean()
        unique_score = min(1, series.nunique() / (len(series) * 0.1))  # Max 10% unique
        variation_score = min(1, series.std() / series.mean() if series.mean() != 0 else 0)
        outlier_score = 1 - outlier_ratio

        # Combinaison pondérée des scores
        return (non_null_score * 0.3 +
                unique_score * 0.2 +
                variation_score * 0.3 +
                outlier_score * 0.2)

    # Identification initiale des types de colonnes
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns

    # Analyse approfondie des colonnes numériques
    numeric_scores = {}
    for col in numeric_cols:
        if is_excluded(col):
            continue

        series = df[col]
        if series.notna().mean() < 0.3:  # Minimum 30% de valeurs non nulles
            continue

        score = calculate_numeric_score(series)
        if score > 0.5:  # Seuil minimal de qualité
            numeric_scores[col] = score

    # Analyse approfondie des colonnes catégorielles
    categorical_scores = {}
    for col in categorical_cols:
        if is_excluded(col):
            continue

        series = df[col]
        n_unique = series.nunique()
        if not (2 <= n_unique <= 50):  # Entre 2 et 50 catégories
            continue

        non_null_ratio = series.notna().mean()
        if non_null_ratio < 0.3:  # Minimum 30% de valeurs non nulles
            continue

        distribution_score = calculate_distribution_score(series.dropna())
        if distribution_score > 0.3:  # Distribution suffisamment équilibrée
            categorical_scores[col] = distribution_score

    # Sélection des meilleures colonnes
    valid_numeric_cols = sorted(numeric_scores.keys(),
                              key=lambda x: numeric_scores[x],
                              reverse=True)[:5]  # Top 5 colonnes numériques

    valid_categorical_cols = sorted(categorical_scores.keys(),
                                  key=lambda x: categorical_scores[x],
                                  reverse=True)[:3]  # Top 3 colonnes catégorielles

    # Si aucune colonne catégorielle n'est trouvée, créer des bins sur les meilleures colonnes numériques
    if not valid_categorical_cols and valid_numeric_cols:
        best_numeric = valid_numeric_cols[0]
        df[f'{best_numeric}_binned'] = pd.qcut(
            df[best_numeric].fillna(df[best_numeric].median()),
            q=min(5, df[best_numeric].nunique()),
            labels=[f'Q{i+1}' for i in range(5)],
            duplicates='drop'
        )
        valid_categorical_cols.append(f'{best_numeric}_binned')

    # Affichage des informations
    print("\nColonnes sélectionnées pour la stratification:")
    for col in valid_categorical_cols:
        if col in categorical_scores:
            print(f"- {col}: {df[col].nunique()} catégories, "
                  f"score={categorical_scores[col]:.3f}")
        else:
            print(f"- {col}: {df[col].nunique()} catégories (binned)")

    print("\nColonnes numériques sélectionnées:")
    for col in valid_numeric_cols:
        print(f"- {col}: score={numeric_scores[col]:.3f}, "
              f"std={df[col].std():.3f}, "
              f"missing={df[col].isna().mean():.1%}")

    return valid_categorical_cols, valid_numeric_cols


def smart_sample(
    df: pd.DataFrame,
    target_size: int | None = None,
    random_state: int = 42,
    verbose: bool = True
) -> tuple[pd.DataFrame, dict[str, dict]]:
    """
    Crée automatiquement un échantillon intelligent du dataset.

    Args:
        df: DataFrame source
        target_size: Taille souhaitée de l'échantillon (défaut: 25% des données)
        random_state: Graine aléatoire pour la reproductibilité
        verbose: Si True, affiche les informations d'analyse

    Returns:
        Tuple (DataFrame échantillonné, métriques et informations)
    """
    if len(df) == 0:
        raise ValueError("Le DataFrame est vide")

    if target_size is None:
        target_size = len(df) // 4

    # Sélection automatique des colonnes
    stratify_cols, numeric_cols = _auto_select_columns(df)

    # Création de la colonne de stratification combinée
    df = df.copy()

    # Si aucune colonne n'est disponible, utiliser un échantillonnage aléatoire
    if not stratify_cols and not numeric_cols:
        sample = df.sample(n=target_size, random_state=random_state)
        metrics = {'sampling_method': 'random', 'reduction_ratio': len(sample) / len(df)}
        return sample, metrics

    # Sinon, utiliser la stratification
    if stratify_cols:
        df['combined_strata'] = df[stratify_cols].astype(str).agg('-'.join, axis=1)
    else:
        # Utiliser des bins sur la première colonne numérique
        first_numeric = numeric_cols[0]
        df['combined_strata'] = pd.qcut(
            df[first_numeric].fillna(df[first_numeric].median()),
            q=min(10, df[first_numeric].nunique()),
            labels=False,
            duplicates='drop'
        )

    # Calcul des proportions cibles pour chaque strate
    strata_counts = df['combined_strata'].value_counts()
    total_samples = len(df)

    # Calcul du nombre d'échantillons par strate
    samples_per_stratum = {}
    for stratum in strata_counts.index:
        stratum_ratio = strata_counts[stratum] / total_samples
        samples_per_stratum[stratum] = max(1, int(target_size * stratum_ratio))

    # Ajustement pour atteindre exactement la taille cible
    total_allocated = sum(samples_per_stratum.values())
    if total_allocated > target_size:
        # Réduire proportionnellement chaque strate
        reduction_ratio = target_size / total_allocated
        for stratum in samples_per_stratum:
            samples_per_stratum[stratum] = max(1, int(samples_per_stratum[stratum] * reduction_ratio))

    # Échantillonnage stratifié avec les tailles ajustées
    sampled_dfs = []
    for stratum, size in samples_per_stratum.items():
        stratum_df = df[df['combined_strata'] == stratum]
        if len(stratum_df) > size:
            sampled_dfs.append(stratum_df.sample(n=size, random_state=random_state))
        else:
            sampled_dfs.append(stratum_df)  # Prendre toute la strate si trop petite

    # Combinaison des échantillons
    sample = pd.concat(sampled_dfs, axis=0)

    # Si l'échantillon est encore trop grand, réduire aléatoirement
    if len(sample) > target_size:
        sample = sample.sample(n=target_size, random_state=random_state)

    # Nettoyage
    sample = sample.drop('combined_strata', axis=1)

    # Calcul des métriques
    metrics = _calculate_metrics(df, sample, stratify_cols, numeric_cols)

    # Visualisations si verbose
    if verbose:
        _plot_distributions(df, sample, stratify_cols, numeric_cols)

    return sample, metrics


def _calculate_metrics(
    original_df: pd.DataFrame,
    sampled_df: pd.DataFrame,
    stratify_cols: list[str],
    numeric_cols: list[str]
) -> dict[str, dict]:
    """
    Calcule les métriques de qualité de l'échantillonnage.

    Métriques calculées :
    1. Ratio de réduction
    2. Pour chaque colonne de stratification :
        - Différence moyenne des proportions
        - Score de Kullback-Leibler (divergence)
        - Test du chi2 d'indépendance
    3. Pour chaque colonne numérique :
        - Différence relative des moyennes
        - Différence relative des écarts-types
        - Test de Kolmogorov-Smirnov
    """
    metrics = {
        'reduction_ratio': len(sampled_df) / len(original_df),
        'columns_used': {
            'stratification': stratify_cols,
            'numerical': numeric_cols
        }
    }

    # Métriques pour les colonnes de stratification
    for col in stratify_cols:
        orig_props = original_df[col].value_counts(normalize=True)
        sample_props = sampled_df[col].value_counts(normalize=True)

        # Différence moyenne des proportions
        diff = np.abs(orig_props - sample_props.reindex(orig_props.index).fillna(0)).mean()
        metrics[f"{col}_prop_difference"] = diff

        # Divergence de Kullback-Leibler (avec lissage pour éviter div/0)
        eps = 1e-10
        kl_div = np.sum(orig_props * np.log((orig_props + eps) / (sample_props.reindex(orig_props.index).fillna(eps) + eps)))
        metrics[f"{col}_kl_divergence"] = kl_div

        # Test du chi2
        orig_counts = original_df[col].value_counts()
        sample_counts = sampled_df[col].value_counts()
        chi2, pval = stats.chi2_contingency([orig_counts, sample_counts])[0:2]
        metrics[f"{col}_chi2_pvalue"] = pval

    # Métriques pour les colonnes numériques
    for col in numeric_cols:
        # Statistiques de base
        orig_mean = original_df[col].mean()
        orig_std = original_df[col].std()

        # Différences relatives
        mean_diff = abs(orig_mean - sampled_df[col].mean()) / (orig_std if orig_std != 0 else 1)
        std_diff = abs(orig_std - sampled_df[col].std()) / (orig_std if orig_std != 0 else 1)

        metrics[f"{col}_mean_difference"] = mean_diff
        metrics[f"{col}_std_difference"] = std_diff

        # Test de Kolmogorov-Smirnov
        ks_stat, ks_pval = stats.ks_2samp(
            original_df[col].dropna(),
            sampled_df[col].dropna()
        )
        metrics[f"{col}_ks_pvalue"] = ks_pval

        # Quartiles
        orig_quant = original_df[col].quantile([0.25, 0.5, 0.75])
        sample_quant = sampled_df[col].quantile([0.25, 0.5, 0.75])
        metrics[f"{col}_quartile_differences"] = {
            'Q1': abs(orig_quant[0.25] - sample_quant[0.25]) / (orig_std if orig_std != 0 else 1),
            'Q2': abs(orig_quant[0.5] - sample_quant[0.5]) / (orig_std if orig_std != 0 else 1),
            'Q3': abs(orig_quant[0.75] - sample_quant[0.75]) / (orig_std if orig_std != 0 else 1)
        }

    return metrics


def _plot_distributions(
    original_df: pd.DataFrame,
    sampled_df: pd.DataFrame,
    stratify_cols: list[str],
    numeric_cols: list[str]
) -> None:
    """
    Génère des visualisations pour comparer les distributions entre l'échantillon
    original et l'échantillon stratifié.
    """
    # Configuration du style
    plt.style.use('default')  # Utilisation du style par défaut de matplotlib

    # Création des sous-graphiques
    n_cols = len(stratify_cols) + len(numeric_cols)
    n_rows = (n_cols + 1) // 2
    fig, axes = plt.subplots(n_rows, 2, figsize=(15, 5*n_rows))
    axes = axes.flatten()

    # Fonction pour calculer le V de Cramer
    def cramers_v(x, y):
        confusion_matrix = pd.crosstab(x, y)
        chi2 = stats.chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum().sum()
        min_dim = min(confusion_matrix.shape) - 1
        return np.sqrt(chi2 / (n * min_dim))

    # Plot des distributions catégorielles
    for i, col in enumerate(stratify_cols):
        ax = axes[i]

        # Calcul des proportions
        orig_props = original_df[col].value_counts(normalize=True)
        samp_props = sampled_df[col].value_counts(normalize=True)

        # Création du graphique en barres
        x = np.arange(len(orig_props))
        width = 0.35

        ax.bar(x - width/2, orig_props.values, width, label='Original', alpha=0.7)
        ax.bar(x + width/2, samp_props.values, width, label='Sampled', alpha=0.7)

        ax.set_title(f'Distribution de {col}')
        ax.set_xticks(x)
        ax.set_xticklabels(orig_props.index, rotation=45, ha='right')
        ax.legend()

        # Ajout du V de Cramer
        v = cramers_v(original_df[col], sampled_df[col])
        ax.text(0.02, 0.98, f"Cramer's V: {v:.3f}",
                transform=ax.transAxes, verticalalignment='top')

    # Plot des distributions numériques
    for i, col in enumerate(numeric_cols):
        ax = axes[i + len(stratify_cols)]

        # Calcul du nombre de bins optimal (limité à 50)
        n_bins = min(50, int(np.sqrt(len(original_df[col].dropna()))))

        # Création des histogrammes avec matplotlib
        ax.hist(original_df[col].dropna(), bins=n_bins, alpha=0.5, label='Original', density=True)
        ax.hist(sampled_df[col].dropna(), bins=n_bins, alpha=0.5, label='Sampled', density=True)

        ax.set_title(f'Distribution de {col}')
        ax.legend()

        # Ajout du test de Kolmogorov-Smirnov
        ks_stat, p_value = stats.ks_2samp(
            original_df[col].dropna(),
            sampled_df[col].dropna()
        )
        ax.text(0.02, 0.98, f"KS p-value: {p_value:.3f}",
                transform=ax.transAxes, verticalalignment='top')

    # Suppression des axes vides si nécessaire
    for i in range(len(stratify_cols) + len(numeric_cols), len(axes)):
        axes[i].remove()

    plt.tight_layout()
    plt.show()
