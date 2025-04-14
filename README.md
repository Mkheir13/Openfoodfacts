# Open Food Facts - Analyse de Données et Clustering

Ce projet fournit une application web Streamlit pour analyser et appliquer des algorithmes de clustering sur les données de produits alimentaires d'Open Food Facts. Il permet aux utilisateurs de téléverser un jeu de données (ou d'utiliser un échantillon), de le prétraiter et d'appliquer divers algorithmes de clustering.

## ✨ Fonctionnalités Principales

*   **Chargement et Prétraitement Interactif :** Chargez vos données CSV ou utilisez un échantillon. Appliquez des étapes de nettoyage et de préparation configurables.
*   **Clustering Avancé :** Appliquez K-Means, DBSCAN, ou GMM avec optimisation des hyperparamètres et sélection des caractéristiques.
*   **Visualisation Intuitive :** Explorez les clusters via des graphiques PCA 2D et 3D, et analysez les caractéristiques distinctives de chaque groupe.
*   **Export et Sauvegarde :** Téléchargez les données prétraitées et sauvegardez les modèles de clustering entraînés.

## 🚀 Installation et Configuration

**Prérequis :**

*   Python (version 3.8+ recommandée)
*   pip (gestionnaire de paquets Python)
*   Git (pour cloner le dépôt)

**Étapes :**

1.  **Cloner le dépôt :**
    ```bash
    git clone <url_du_depot>
    cd <repertoire_du_projet>
    ```

2.  **Créer un Environnement Virtuel et Installer les Dépendances :**
    Il est fortement recommandé d'utiliser un environnement virtuel pour isoler les dépendances.
    *   **Création :**
        ```bash
        python -m venv .venv
        ```
    *   **Activation :**
        *   Windows (PowerShell/CMD) : `.\.venv\Scripts\activate`
        *   Linux/macOS (bash/zsh) : `source .venv/bin/activate`
        *(L'invite de commande indiquera que l'environnement `.venv` est actif)*
    *   **Installation des dépendances :**
        ```bash
        pip install -r requirements.txt
        ```
    *   **Désactivation (quand terminé) :** `deactivate`
    *(N'oubliez pas de réactiver l'environnement pour utiliser le projet)*

3.  **Préparer les Données :**
    *   **Option A : Télécharger le jeu de données complet :** Le fichier `config.yaml` pointe vers le jeu de données anglais (`https://static.openfoodfacts.org/data/en.openfoodfacts.org.products.csv`). Téléchargez-le et placez-le dans un dossier `data/`.
    *   **Option B : Créer un échantillon :** Pour un démarrage rapide, utilisez le script `create_reduced_file.py` sur le fichier complet téléchargé.
        ```bash
        # Crée data/reduced/full_datas_100000.csv (100 000 lignes)
        python create_reduced_file.py data/en.openfoodfacts.org.products.csv 100000
        ```
        L'application proposera d'utiliser cet échantillon s'il est trouvé.

## ▶️ Utilisation de l'Application Streamlit

1.  **Lancer l'application :**
    Assurez-vous que votre environnement virtuel est activé, puis exécutez :
    ```bash
    streamlit run app.py
    ```
    L'application s'ouvrira dans votre navigateur web (généralement `http://localhost:8501`).

2.  **Interface Utilisateur :**
    L'application est divisée en deux onglets principaux :
    *   `📊 Preprocessing`
    *   `🧠 Clustering`

3.  **Onglet `📊 Preprocessing` :**
    *   **Chargement :**
        *   Utilisez le bouton "Browse files" pour téléverser votre propre fichier CSV (séparateur tabulation `\t` attendu).
        *   Si un fichier échantillon (`data/reduced/full_datas_100000.csv`) existe, un bouton "Utiliser le fichier échantillon par défaut" apparaîtra. Cliquez dessus pour le charger.
        *   Un aperçu (`.head()`) des données chargées s'affiche.
    *   **Nettoyage et Prétraitement :**
        *   Ajustez les paramètres de prétraitement dans la barre latérale (Sidebar) si nécessaire :
            *   *Nombre max de catégories* : Pour la conversion des variables catégorielles.
            *   *Ratio min de valeurs uniques* : Pour la sélection de caractéristiques.
        *   Cliquez sur le bouton "Lancer le Preprocessing".
        *   Le processus effectue plusieurs étapes (extraction numérique, préparation des caractéristiques pour le clustering) avec des indicateurs de progression (`spinner`).
        *   Un **Résumé du Nettoyage** s'affiche (dimensions avant/après, colonnes/lignes supprimées).
        *   **Sauvegarder les données :** Les données prétraitées sont automatiquement sauvegardées dans `data/preprocessed/` (ex: `full_datas_100000_preprocessed.csv`).
        *   Un bouton **"Télécharger le CSV Prétraité"** apparaît pour récupérer ce fichier.
    *   Un aperçu des données prétraitées est affiché. Vous pouvez maintenant passer à l'onglet Clustering.

4.  **Onglet `🧠 Clustering` :**
    *   **Chargement des données prétraitées :**
        *   Si vous venez de terminer le prétraitement dans l'onglet précédent, les données sont déjà chargées.
        *   Sinon, un avertissement apparaît. Vous pouvez téléverser un fichier CSV **prétraité** existant via le bouton "Browse files".
    *   **Configuration :**
        *   **Sélection des Colonnes :** Choisissez les colonnes numériques (préalablement nettoyées et mises à l'échelle implicitement par la suite) à utiliser pour le clustering dans la liste déroulante `multiselect`.
        *   **Choix de l'Algorithme :** Sélectionnez "K-Means", "DBSCAN", ou "GMM" dans la liste déroulante `selectbox`.
        *   **Configuration de l'Algorithme :** Des options spécifiques apparaissent en fonction de l'algorithme choisi :
            *   **K-Means/GMM :**
                *   *Optimisation du nombre de clusters/composantes :* Définissez une plage (`min`/`max`) et une méthode d'évaluation (`scoring_method` : silhouette, etc.). Cochez/décochez la case pour activer/désactiver la recherche automatique du nombre optimal.
                *   *Configuration de l'entraînement :* Définissez le nombre de clusters/composantes (si l'optimisation est désactivée), la méthode d'optimisation des paramètres (`optimize_method`), et d'autres hyperparamètres (max_iter, init, etc.).
                *   *Sauvegarde :* Cochez la case pour sauvegarder le modèle entraîné et spécifiez le chemin (`.pkl`).
            *   **DBSCAN :**
                *   Ajustez `eps` et `min_samples` via les sliders.
    *   **Lancer le Clustering :** Cliquez sur le bouton "Lancer le Clustering [Nom de l'Algo]".
    *   **Résultats :**
        *   Le temps d'entraînement est affiché.
        *   Les métriques de performance (Score Silhouette, Calinski-Harabasz, Davies-Bouldin) sont présentées.
        *   Le nombre de clusters trouvés (et de points de bruit pour DBSCAN) est indiqué.
        *   **Visualisations :** Des graphiques PCA en 2D et 3D montrent la répartition des points colorés par cluster.
        *   **Aperçu des Données :** Un tableau (`dataframe`) montre les premières lignes des données avec la colonne `cluster` assignée.
        *   **Analyse des Clusters :** Pour chaque cluster trouvé, les caractéristiques ayant les moyennes les plus élevées et les plus distinctives (par rapport à la moyenne globale) sont listées pour aider à interpréter les groupes.

## 📂 Structure du Projet

```
├── .venv/           # Environnement virtuel Python (si créé ici)
├── .streamlit/      # Configuration Streamlit (si existante)
├── data/            # Fichiers de données (brutes, réduites, prétraitées)
│   ├── reduced/       # Jeux de données échantillonnés
│   └── preprocessed/  # Jeux de données prétraités sauvegardés
├── models/          # Modèles de clustering sauvegardés
├── notebooks/       # Notebooks Jupyter pour l'exploration (si existants)
├── results/         # Résultats d'analyse (si sauvegardés)
├── scripts/         # Modules Python principaux
│   ├── data/        # Scripts de chargement, nettoyage, échantillonnage des données
│   ├── features/    # Scripts d'ingénierie des caractéristiques (scaling, etc.)
│   └── models/      # Implémentations des algorithmes de clustering
├── tests/           # Tests unitaires et d'intégration
├── .gitignore       # Fichiers/dossiers ignorés par Git
├── app.py           # Fichier principal de l'application Streamlit
├── config.yaml      # Configuration (ex: chemins des données)
├── create_reduced_file.py # Script pour créer des échantillons de données
├── MANIFEST.in      # Fichier manifeste pour le packaging
├── README.md        # Ce fichier
├── requirements.txt # Dépendances Python
├── setup.py         # Script pour packager le projet
└── ...              # Autres fichiers de configuration
```
