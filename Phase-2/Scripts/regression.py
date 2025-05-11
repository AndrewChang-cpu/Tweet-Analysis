#!/usr/bin/env python3
"""
Linear Regression on Augmented Tweet Data

Usage:
  python regression_model.py --input_dir path/to/tweets_augmented
"""

import argparse
import os
import glob
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir", required=True,
        help="Directory where tweets_augmented CSV files were written"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    csv_files = glob.glob(os.path.join(args.input_dir, "tweets_augmented", "*.csv"))
    pdf = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)

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

    print(X_train.shape)
    print(X_train.head())

    model = LinearRegression().fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\nRegression results:")
    print(f"R2: {r2_score(y_test, y_pred):.3f}")
    print(f"RMSE: {mean_squared_error(y_test, y_pred, squared=False):.3f}")
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.3f}")

    coef_df = pd.DataFrame({"feature": features.columns, "coef": model.coef_})
    print("\nCoefficients:")
    print(coef_df)

if __name__ == "__main__":
    main()
