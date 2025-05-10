import org.apache.spark.sql.{SparkSession, DataFrame}
import org.apache.spark.sql.functions._

/**
 * ParquetTweetExtraction.scala
 *
 * Reads raw tweet JSON, extracts key features, and writes a Parquet dataset for downstream Ray processing.
 * Usage:
 *   spark-submit \
 *     --class ParquetTweetExtraction \
 *     path/to/jar.jar \
 *     <input_json_path> <output_parquet_path>
 */
object FabricTweetParquetExtraction {
  def main(args: Array[String]): Unit = {
    if (args.length != 2) {
      System.err.println("Usage: FabricTweetParquetExtraction <input_json> <output_parquet_path>")
      System.exit(1)
    }
    val Array(inputJson, outputParquet) = args

    // 1. Initialize SparkSession
    val spark = SparkSession.builder()
      .appName("FabricTweetParquetExtraction")
      .getOrCreate()
    import spark.implicits._

    // 2. Read raw JSON tweets (one JSON object per line)
    val rawDF = spark.read
      .option("mode", "PERMISSIVE")
      .json(inputJson)

    // 3. Extract only required fields
    val featuresDF = rawDF.select(
      col("full_text"),
      col("retweet_count"),
      col("favorite_count"),
      size(col("entities.hashtags")).alias("num_hashtags"),
      length(col("full_text")).alias("text_length"),
      col("user.followers_count").alias("user_followers"),
      col("user.friends_count").alias("user_friends")
    )

    // 4. Write out as Parquet (with coalesce to limit file count)
    featuresDF.coalesce(4)
      .write
      .mode("overwrite")
      .parquet(outputParquet)

    // 5. Stop Spark
    spark.stop()
  }
}