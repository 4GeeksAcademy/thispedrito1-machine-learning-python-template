from pathlib import Path

import pandas as pd
from transformers import pipeline


MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"
MODEL_REVISION = "8f6f4e3a8f70be4b65d3a4a8762b6d781cda240d"
DEFAULT_INPUT_PATH = Path(__file__).parents[1] / "data" / "raw" / "reviews.csv"
DEFAULT_OUTPUT_PATH = (
	Path(__file__).parents[1] / "data" / "processed" / "reviews_with_sentiment.csv"
)


def sentiment_band(stars: int) -> str:
	"""Map the model's five-star output to the business-facing bands."""
	if stars <= 2:
		return "negative"
	if stars == 3:
		return "neutral"
	return "positive"


def enrich_reviews(
	input_path: Path = DEFAULT_INPUT_PATH,
	output_path: Path = DEFAULT_OUTPUT_PATH,
	batch_size: int = 16,
) -> pd.DataFrame:
	"""Classify every review once and persist the enriched dataset."""
	reviews = pd.read_csv(input_path)
	required_columns = {"review_id", "rating", "review_text"}
	missing_columns = required_columns.difference(reviews.columns)
	if missing_columns:
		raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

	classifier = pipeline(
		"sentiment-analysis",
		model=MODEL_NAME,
		revision=MODEL_REVISION,
		truncation=True,
	)
	predictions = classifier(
		reviews["review_text"].astype(str).tolist(),
		batch_size=batch_size,
	)

	enriched = reviews.copy()
	enriched["predicted_stars"] = [int(item["label"].split()[0]) for item in predictions]
	enriched["prediction_score"] = [item["score"] for item in predictions]
	enriched["sentiment_band"] = enriched["predicted_stars"].map(sentiment_band)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	enriched.to_csv(output_path, index=False)
	return enriched


if __name__ == "__main__":
	result = enrich_reviews()
	print(f"Processed {len(result)} reviews")
	print(result["sentiment_band"].value_counts().sort_index().to_string())
