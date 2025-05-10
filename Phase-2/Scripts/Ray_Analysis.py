#!/usr/bin/env python3
"""
Ray Analysis Pipeline (Python)

This script performs the Ray-based portion of the project:
1. Load the CSV produced by the Scala extraction pipeline (7 specific columns)
2. Augment with sentiment and emotion via HuggingFace models in Ray
3. Run descriptive analytics
4. Train and evaluate a multiple linear regression

Usage:
  python ray_analysis.py --input_csv path/to/output_tweets_features/*.csv --output_dir path/to/output_dir
"""
import argparse
import ray
from ray.data import read_csv
from transformers import pipeline
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_csv", required=True,
        help="Glob path to extracted features CSVs (only 7 columns should be present)"
    )
    parser.add_argument(
        "--output_dir", default="tweets_augmented",
        help="Directory to write augmented CSV and results"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ray.init()

    # 1) Load CSV as Ray Dataset with proper quoting and select only the 7 columns
    ds = read_csv(args.input_csv)

    # 2) NLP augmentation
    sentiment_pipe = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=-1
    )
    emotion_pipe = pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        return_all_scores=True,
        device=-1
    )

    def add_nlp(batch: pd.DataFrame) -> pd.DataFrame:
        texts = batch["full_text"].tolist()
        # sentiment
        sents = sentiment_pipe(texts, truncation=True)
        batch["sentiment_label"] = [s["label"] for s in sents]
        batch["sentiment_score"] = [s["score"] for s in sents]
        # emotion
        emos = emotion_pipe(texts, truncation=True)
        batch["emotion_label"] = [max(e, key=lambda x: x["score"])["label"] for e in emos]
        batch["emotion_score"] = [max(e, key=lambda x: x["score"])["score"] for e in emos]
        # popularity score
        batch["popularity_score"] = batch["retweet_count"] + batch["favorite_count"]
        return batch

    ds = ds.map_batches(
        add_nlp,
        batch_size=256,
        batch_format="pandas"
    )

    # 3) Persist augmented dataset
    ds.write_csv(f"{args.output_dir}/tweets_augmented", header=True)

    # 4) Convert to pandas for analytics/modeling
    pdf = ds.to_pandas()

    # 5) Descriptive analytics
    print("Avg popularity by sentiment:")
    print(pdf.groupby("sentiment_label")["popularity_score"].mean())

    print("\nAvg popularity by emotion:")
    print(pdf.groupby("emotion_label")["popularity_score"].mean())

    print("\nAvg popularity with/without hashtags:")
    print(pdf.groupby(pdf["num_hashtags"]>0)["popularity_score"].mean())

    print(f"\nText length correlation: {pdf['text_length'].corr(pdf['popularity_score']):.3f}")

    # 6) Regression modeling
    pdf["has_hashtag"] = (pdf["num_hashtags"] > 0).astype(int)
    pdf["has_media"] = (pdf.get("num_media", pd.Series(0)) > 0).astype(int)

    features = pdf[[
        "sentiment_score",
        "emotion_score",
        "text_length",
        "has_hashtag",
        "has_media",
        "user_followers",
        "user_friends"
    ]]

    target = pdf["popularity_score"]

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )
    model = LinearRegression().fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\nRegression results:")
    print(f"R2: {r2_score(y_test, y_pred):.3f}")
    print(f"RMSE: {mean_squared_error(y_test, y_pred, squared=False):.3f}")
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.3f}")

    # Coefficients
    coef_df = pd.DataFrame({"feature": features.columns, "coef": model.coef_})
    print("\nCoefficients:")
    print(coef_df)

    ray.shutdown()

if __name__ == "__main__":
    main()
