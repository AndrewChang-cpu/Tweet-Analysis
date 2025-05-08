from pyspark.sql import SparkSession
import json

# 1. Create SparkSession (Fabric will pick up your workspace/account settings automatically)
spark = SparkSession.builder \
    .appName("FabricTweetRDDExtraction") \
    .getOrCreate()

sc = spark.sparkContext

# 2. Point to your raw JSON file(s) in your Lakehouse/Lake:
#    e.g. "abfss://<container>@<storage-account>.dfs.core.windows.net/path/to/tweets.json"
input_path = "abfss://<container>@<storage-account>.dfs.core.windows.net/path/to/tweets.json"

# 3. Read as text and parse JSON into Python dicts
tweets_rdd = sc.textFile(input_path) \
    .map(lambda line: json.loads(line)) \
    .map(lambda t: (
        # 1) full_text
        t.get("full_text", ""),
        # 2) retweet_count
        t.get("retweet_count", 0),
        # 3) favorite_count
        t.get("favorite_count", 0),
        # 4) num_hashtags
        len(t.get("entities", {}).get("hashtags", [])),
        # 5) text_length
        len(t.get("full_text", "")),
        # 6) user.followers_count
        t.get("user", {}).get("followers_count", 0),
        # 7) user.friends_count
        t.get("user", {}).get("friends_count", 0)
    ))

# 4. (Optional) inspect a few records
for record in tweets_rdd.take(5):
    print(record)

# 5. Now you can save this RDD back as Parquet or convert to DataFrame:
#    a) Convert to DataFrame with column names:
columns = [
    "full_text",
    "retweet_count",
    "favorite_count",
    "num_hashtags",
    "text_length",
    "user_followers",
    "user_friends"
]
tweets_df = tweets_rdd.toDF(columns)

#    b) Write out as Parquet for downstream Ray processing
tweets_df.write.mode("overwrite").parquet("abfss://<container>@<storage-account>.dfs.core.windows.net/processed/tweets_features.parquet")

# 6. Stop the session if running as a standalone script
spark.stop()
