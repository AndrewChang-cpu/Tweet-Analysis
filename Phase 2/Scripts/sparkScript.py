from pyspark.sql import SparkSession
from pyspark.sql.functions import col, size, length, to_date, current_date, datediff

spark = SparkSession.builder.appName("TweetFeatures").getOrCreate()
tweets = spark.read.json("out.json")

df = tweets.select(
    col("full_text").alias("text"),
    col("retweet_count"),
    col("favorite_count"),
    size(col("entities.hashtags")).alias("num_hashtags"),
    (size(col("entities.hashtags")) > 0).alias("has_hashtag"),
    size(col("entities.media")).alias("num_media"),
    (size(col("entities.media")) > 0).alias("has_media"),
    length(col("full_text")).alias("text_length"),
    col("user.followers_count").alias("user_followers"),
    col("user.friends_count").alias("user_friends"),
    col("user.statuses_count").alias("user_statuses"),
    col("user.verified").alias("user_verified"),
    datediff(current_date(), to_date(col("user.created_at"), "EEE MMM dd HH:mm:ss Z yyyy")
            ).alias("account_age_days")
)

# Write out for your Ray NLP step
df.write.mode("overwrite").parquet("path/to/cleaned_tweets.parquet")
