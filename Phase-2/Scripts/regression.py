import argparse
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import seaborn as sns

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir", required=True,
        help="Directory where tweets_augmented CSV files were written"
    )
    return parser.parse_args()

def run_regression(pdf: pd.DataFrame):
    pdf["has_hashtag"] = (pdf["num_hashtags"] > 0).astype(int)
    pdf["has_media"] = (pdf.get("num_media", pd.Series(0)) > 0).astype(int)
    pdf = pdf.drop(columns=["has_media"], errors="ignore")

    features = pdf[[ 
        "sentiment_score",
        "emotion_score",
        "text_length",
        "has_hashtag",
        "user_followers",
        "user_friends"
    ]]
    target = pdf["popularity_score"]

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )

    print('Dimensions', X_train.shape)
    print(X_train[:5])
    model = LinearRegression().fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\nRegression results:")
    print(f"R2: {r2_score(y_test, y_pred):.3f}")
    print(f"RMSE: {mean_squared_error(y_test, y_pred, squared=False):.3f}")
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.3f}")

    coef_df = pd.DataFrame({"feature": features.columns, "coef": model.coef_})
    print("\nCoefficients:")
    print(coef_df)

def main():
    args = parse_args()
    csv_files = glob.glob(os.path.join(args.input_dir, "tweets_augmented", "*.csv"))
    pdf = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
    
    # Rescale sentiment_score between 0 and 1
    pdf["sentiment_score"] = pdf.apply(
        lambda row: 1 - row["sentiment_score"] if row.get("sentiment_label", "").lower() == "negative" else row["sentiment_score"],
        axis=1
    )
    pdf["has_hashtag"] = (pdf["num_hashtags"] > 0).astype(int)
    pdf["has_media"] = (pdf.get("num_media", pd.Series(0)) > 0).astype(int)

    # Plots directory
    plots_dir = "visualizations"
    os.makedirs(plots_dir, exist_ok=True)

    def scatterplot(x, y, xlabel, ylabel, title, filename):
        plt.figure(figsize=(8, 5))
        plt.scatter(x, y, alpha=0.4)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, filename))
        plt.close()

    # Scatter plots vs popularity
    scatterplot(pdf["sentiment_score"], pdf["popularity_score"], "Sentiment Score", "Popularity", "Sentiment vs Popularity", "sentiment_vs_popularity.png")
    scatterplot(pdf["sentiment_score"], pdf["favorite_count"], "Sentiment Score", "Favorites", "Sentiment vs Favorites", "sentiment_vs_favorites.png")
    scatterplot(pdf["sentiment_score"], pdf["retweet_count"], "Sentiment Score", "Retweets", "Sentiment vs Retweets", "sentiment_vs_retweets.png")
    scatterplot(pdf["num_hashtags"], pdf["popularity_score"], "Number of Hashtags", "Popularity", "Hashtags vs Popularity", "hashtags_vs_popularity.png")
    scatterplot(pdf["text_length"], pdf["popularity_score"], "Text Length", "Popularity", "Text Length vs Popularity", "length_vs_popularity.png")
    scatterplot(pdf["user_followers"], pdf["popularity_score"], "Followers", "Popularity", "Followers vs Popularity", "followers_vs_popularity.png")
        

    # Scatter plots vs favorites and retweets
    scatterplot(pdf["text_length"], pdf["favorite_count"], "Text Length", "Favorites", "Text Length vs Favorites", "length_vs_favorites.png")
    scatterplot(pdf["text_length"], pdf["retweet_count"], "Text Length", "Retweets", "Text Length vs Retweets", "length_vs_retweets.png")
    scatterplot(pdf["num_hashtags"], pdf["favorite_count"], "Number of Hashtags", "Favorites", "Hashtags vs Favorites", "hashtags_vs_favorites.png")
    scatterplot(pdf["num_hashtags"], pdf["retweet_count"], "Number of Hashtags", "Retweets", "Hashtags vs Retweets", "hashtags_vs_retweets.png")
    scatterplot(pdf["user_followers"], pdf["favorite_count"], "Followers", "Favorites", "Followers vs Favorites", "followers_vs_favorites.png")
    scatterplot(pdf["user_followers"], pdf["retweet_count"], "Followers", "Retweets", "Followers vs Retweets", "followers_vs_retweets.png")

    # Boxplot: Emotion vs popularity
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="emotion_label", y="popularity_score", data=pdf)

    # Compute descriptive stats
    emotion_stats = pdf.groupby("emotion_label")["popularity_score"].describe()[["25%", "50%", "75%", "count", "max"]]

    # Add labels above the boxes
    for i, emotion in enumerate(emotion_stats.index):
        stats = emotion_stats.loc[emotion]
        label = (
            f"Q1: {stats['25%']:.1f}\n"
            f"Q2: {stats['50%']:.1f}\n"
            f"Q3: {stats['75%']:.1f}\n"
            f"N: {int(stats['count'])}"
        )
        y_pos = stats["75%"] + 1 * stats["max"]  # 5% above Q3 for spacing
        plt.text(i, y_pos, label, ha='center', fontsize=9, color="black")

    plt.title("Popularity Score by Emotion")
    plt.xlabel("Emotion")
    plt.ylabel("Popularity Score")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "emotion_vs_popularity_boxplot.png"))
    plt.close()

    # Run Regression
    run_regression(pdf)


if __name__ == "__main__":
    main()
