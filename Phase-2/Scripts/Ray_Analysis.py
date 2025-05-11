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
import os
import ray
from ray.data import read_csv
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from glob import glob
import time
from transformers import pipeline

# Download once and cache locally
print("Downloading models to local cache...")
pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)
print("Download complete.")

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

@ray.remote(num_gpus=1)
def run_batch(batch: pd.DataFrame) -> pd.DataFrame:
    from transformers import pipeline
    import torch

    device = 0 if torch.cuda.is_available() else -1
    print("IS GPU AVAILABLE?", torch.cuda.is_available())

    sentiment_pipe = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=device,
        local_files_only=True
    )
    emotion_pipe = pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        return_all_scores=True,
        device=device,
        local_files_only=True
    )

    texts = batch["full_text"].tolist()
    sents = sentiment_pipe(texts, truncation=True)
    emos = emotion_pipe(texts, truncation=True)
    batch["sentiment_label"] = [s["label"] for s in sents]
    batch["sentiment_score"] = [s["score"] for s in sents]
    batch["emotion_label"] = [max(e, key=lambda x: x["score"])["label"] for e in emos]
    batch["emotion_score"] = [max(e, key=lambda x: x["score"])["score"] for e in emos]
    batch["popularity_score"] = batch["retweet_count"] + batch["favorite_count"]
    return batch

def main():
    args = parse_args()
    ray.init()

    input_path = args.input_csv
    csv_files = glob(os.path.join(input_path, "*.csv")) if os.path.isdir(input_path) else glob(input_path)
    ds = read_csv(csv_files)

    start_time = time.time()
    ds = ds.map_batches(
        lambda b: ray.get(run_batch.remote(b)),
        batch_size=256,
        batch_format="pandas"
    )
    end_time = time.time()

    print(f"Inference time: {end_time - start_time:.2f} seconds")

    ds.write_csv(f"{args.output_dir}/tweets_augmented")

    pdf = ds.to_pandas()

    print("Avg popularity by sentiment:")
    print(pdf.groupby("sentiment_label")["popularity_score"].mean())

    print("\nAvg popularity by emotion:")
    print(pdf.groupby("emotion_label")["popularity_score"].mean())

    print("\nAvg popularity with/without hashtags:")
    print(pdf.groupby(pdf["num_hashtags"] > 0)["popularity_score"].mean())

    print(f"\nText length correlation: {pdf['text_length'].corr(pdf['popularity_score']):.3f}")

    ray.shutdown()

if __name__ == "__main__":
    main()

