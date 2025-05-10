<<<<<<< HEAD
import org.apache.spark.sql.{Row}
import org.apache.spark.sql.types._
import scala.util.parsing.json.JSON

val inputPath = "out.json"

val tweetsRDD = sc.textFile(inputPath).map(line => JSON.parseFull(line).getOrElse(Map.empty[String, Any])).map {
  case t: Map[String, Any] =>
    val rawText = t.getOrElse("full_text", "").toString
    val cleanText = rawText.replaceAll("[,\n\r]", " ")
    val retweetCount = t.get("retweet_count").map(_.asInstanceOf[Double].toInt).getOrElse(0)
    val favoriteCount = t.get("favorite_count").map(_.asInstanceOf[Double].toInt).getOrElse(0)
    val entities = t.get("entities").getOrElse(Map.empty).asInstanceOf[Map[String, Any]]
    val hashtags = entities.get("hashtags").map(_.asInstanceOf[List[Any]]).getOrElse(List.empty)
    val numHashtags = hashtags.size
    val textLength = cleanText.length
    val user = t.get("user").getOrElse(Map.empty).asInstanceOf[Map[String, Any]]
    val followersCount = user.get("followers_count").map(_.asInstanceOf[Double].toInt).getOrElse(0)
    val friendsCount = user.get("friends_count").map(_.asInstanceOf[Double].toInt).getOrElse(0)
    Row(cleanText, retweetCount, favoriteCount, numHashtags, textLength, followersCount, friendsCount)
  case _ => Row("", 0, 0, 0, 0, 0, 0)
=======
// import org.apache.spark.sql.{SparkSession, Row}
// import org.apache.spark.sql.types._
// import scala.util.parsing.json.JSON

// object FabricTweetRDDExtraction {
//   def main(args: Array[String]): Unit = {
//     // 1. Create SparkSession
//     val spark = SparkSession.builder()
//       .appName("FabricTweetRDDExtraction")
//       .getOrCreate()

//     val sc = spark.sparkContext
//     val inputPath = "out.json"

//     // 2. Read raw lines and parse JSON
//     val tweetsRDD = sc.textFile(inputPath).map(line => JSON.parseFull(line).getOrElse(Map.empty[String, Any])).map {
//       case t: Map[String, Any] =>
//         val fullText = t.getOrElse("full_text", "").toString
//         val retweetCount = t.get("retweet_count").map(_.asInstanceOf[Double].toInt).getOrElse(0)
//         val favoriteCount = t.get("favorite_count").map(_.asInstanceOf[Double].toInt).getOrElse(0)
//         val entities = t.get("entities").getOrElse(Map.empty).asInstanceOf[Map[String, Any]]
//         val hashtags = entities.get("hashtags").map(_.asInstanceOf[List[Any]]).getOrElse(List.empty)
//         val numHashtags = hashtags.size
//         val textLength = fullText.length
//         val user = t.get("user").getOrElse(Map.empty).asInstanceOf[Map[String, Any]]
//         val followersCount = user.get("followers_count").map(_.asInstanceOf[Double].toInt).getOrElse(0)
//         val friendsCount = user.get("friends_count").map(_.asInstanceOf[Double].toInt).getOrElse(0)
//         Row(fullText, retweetCount, favoriteCount, numHashtags, textLength, followersCount, friendsCount)
//       case _ => Row("", 0, 0, 0, 0, 0, 0)
//     }

//     // 3. Define schema
//     val schema = StructType(List(
//       StructField("full_text", StringType, true),
//       StructField("retweet_count", IntegerType, true),
//       StructField("favorite_count", IntegerType, true),
//       StructField("num_hashtags", IntegerType, true),
//       StructField("text_length", IntegerType, true),
//       StructField("user_followers", IntegerType, true),
//       StructField("user_friends", IntegerType, true)
//     ))

//     // 4. Convert RDD to DataFrame
//     val tweetsDF = spark.createDataFrame(tweetsRDD, schema)

//     // 5. Write to CSV
//     tweetsDF.write.mode("overwrite").option("header", "true").csv("output_tweets_features")

//     // 6. Stop session
//     spark.stop()
//   }
// }


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
>>>>>>> d38a488 (change csv to parquet scala)
}

val schema = StructType(List(
  StructField("full_text", StringType, true),
  StructField("retweet_count", IntegerType, true),
  StructField("favorite_count", IntegerType, true),
  StructField("num_hashtags", IntegerType, true),
  StructField("text_length", IntegerType, true),
  StructField("user_followers", IntegerType, true),
  StructField("user_friends", IntegerType, true)
))

val tweetsDF = spark.createDataFrame(tweetsRDD, schema)

// Write as single CSV file
tweetsDF.coalesce(1).write.mode("overwrite").option("header", "true").csv("output_tweets_features.csv")
