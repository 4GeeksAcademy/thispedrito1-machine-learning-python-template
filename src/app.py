from pathlib import Path
from urllib.request import urlretrieve

import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).parents[1]
DATA_URL = "https://raw.githubusercontent.com/4GeeksAcademy/k-means-project-tutorial/main/housing.csv"
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "housing.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed"
MODELS_PATH = PROJECT_ROOT / "models"
FEATURE_COLUMNS = ["Latitude", "Longitude", "MedInc"]
RANDOM_STATE = 42
N_CLUSTERS = 6


def load_housing_data(input_path: Path = RAW_DATA_PATH) -> pd.DataFrame:
	"""Load the housing dataset, downloading it when it is not local."""
	input_path.parent.mkdir(parents=True, exist_ok=True)
	if not input_path.exists():
		urlretrieve(DATA_URL, input_path)

	data = pd.read_csv(input_path)
	missing_columns = set(FEATURE_COLUMNS).difference(data.columns)
	if missing_columns:
		raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

	features = data[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce").dropna()
	if features.empty:
		raise ValueError("The dataset has no valid rows after cleaning")
	return features.reset_index(drop=True)


def build_kmeans_pipeline() -> Pipeline:
	return Pipeline(
		[
			("scaler", StandardScaler()),
			(
				"kmeans",
				KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10),
			),
		]
	)


def build_classifier_pipeline() -> Pipeline:
	return Pipeline(
		[
			("scaler", StandardScaler()),
			(
				"classifier",
				RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
			),
		]
	)


def run_pipeline() -> dict:
	data = load_housing_data()
	train_data, test_data = train_test_split(
		data, test_size=0.2, random_state=RANDOM_STATE
	)

	kmeans_pipeline = build_kmeans_pipeline()
	train_clusters = kmeans_pipeline.fit_predict(train_data[FEATURE_COLUMNS])
	test_clusters = kmeans_pipeline.predict(test_data[FEATURE_COLUMNS])

	train_output = train_data.copy()
	test_output = test_data.copy()
	train_output["cluster"] = train_clusters
	test_output["cluster"] = test_clusters

	classifier_pipeline = build_classifier_pipeline()
	classifier_pipeline.fit(train_data[FEATURE_COLUMNS], train_clusters)
	classification_metrics = classification_report(
		test_clusters,
		classifier_pipeline.predict(test_data[FEATURE_COLUMNS]),
		output_dict=True,
		zero_division=0,
	)

	PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)
	MODELS_PATH.mkdir(parents=True, exist_ok=True)
	train_output.to_csv(PROCESSED_DATA_PATH / "train.csv", index=False)
	test_output.to_csv(PROCESSED_DATA_PATH / "test.csv", index=False)
	joblib.dump(kmeans_pipeline, MODELS_PATH / "kmeans_model.joblib")
	joblib.dump(classifier_pipeline, MODELS_PATH / "supervised_classifier.joblib")

	scaled_train = kmeans_pipeline.named_steps["scaler"].transform(
		train_data[FEATURE_COLUMNS]
	)
	return {
		"rows": len(data),
		"train_rows": len(train_data),
		"test_rows": len(test_data),
		"inertia": kmeans_pipeline.named_steps["kmeans"].inertia_,
		"silhouette_score": silhouette_score(scaled_train, train_clusters),
		"classification_metrics": classification_metrics,
	}


if __name__ == "__main__":
	results = run_pipeline()
	print(f"Rows: {results['rows']} (train={results['train_rows']}, test={results['test_rows']})")
	print(f"K-Means inertia: {results['inertia']:.2f}")
	print(f"Train silhouette score: {results['silhouette_score']:.3f}")
	print(
		f"Classifier accuracy: "
		f"{results['classification_metrics']['accuracy']:.3f}"
	)
