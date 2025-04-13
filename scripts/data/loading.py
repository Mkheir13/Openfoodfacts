"""
Module pour le chargement des données OpenFoodFacts.
"""
import pandas as pd
from typing import Optional

def load_openfoodfacts(n_samples: Optional[int] = None) -> pd.DataFrame:
	"""
	Charge les données depuis l'API OpenFoodFacts.
	
	Args:
		n_samples: Nombre d'échantillons à charger. Si None, charge tout le dataset.
	
	Returns:
		pd.DataFrame: DataFrame contenant les données OpenFoodFacts
	"""
	# Chargement des données depuis le fichier CSV
	url = "data/reduced/full_datas_100000.csv"
	
	# Chargement des données avec pandas
	df = pd.read_csv(
		url,
		sep='\t',
		encoding="utf-8",
		nrows=n_samples
	)
	
	print(f"Dataset chargé : {len(df)} lignes, {len(df.columns)} colonnes")
	
	return df 