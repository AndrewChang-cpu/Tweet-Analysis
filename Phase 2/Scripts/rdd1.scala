import org.apache.spark.sql.{SparkSession, Row}
import org.apache.spark.sql.types._
import scala.util.parsing.json.JSON

object FabricTweetRDDExtraction {
  def main(args: Array[String]): Unit = {
    // 1. Create SparkSession
    val spark = SparkSession.builder()
      .appName("FabricTweetRDDExtraction")
      .getOrCreate()

    val sc = spark.sparkContext
    val inputPath = "out.json"

    // 2. Read raw lines and parse JSON
    val tweetsRDD = sc.textFile(inputPath).map(line => JSON.parseFull(line).getOrElse(Map.empty[String, Any])).map {
      case t: Map[String, Any] =>
        val fullText = t.getOrElse("full_text", "").toString
        val retweetCount = t.get("retweet_count").map(_.asInstanceOf[Double].toInt).getOrElse(0)
        val favoriteCount = t.get("favorite_count").map(_.asInstanceOf[Double].toInt).getOrElse(0)
        val entities = t.get("entities").getOrElse(Map.empty).asInstanceOf[Map[String, Any]]
        val hashtags = entities.get("hashtags").map(_.asInstanceOf[List[Any]]).getOrElse(List.empty)
        val numHashtags = hashtags.size
        val textLength = fullText.length
        val user = t.get("user").getOrElse(Map.empty).asInstanceOf[Map[String, Any]]
        val followersCount = user.get("followers_count").map(_.asInstanceOf[Double].toInt).getOrElse(0)
        val friendsCount = user.get("friends_count").map(_.asInstanceOf[Double].toInt).getOrElse(0)
        Row(fullText, retweetCount, favoriteCount, numHashtags, textLength, followersCount, friendsCount)
      case _ => Row("", 0, 0, 0, 0, 0, 0)
    }

    // 3. Define schema
    val schema = StructType(List(
      StructField("full_text", StringType, true),
      StructField("retweet_count", IntegerType, true),
      StructField("favorite_count", IntegerType, true),
      StructField("num_hashtags", IntegerType, true),
      StructField("text_length", IntegerType, true),
      StructField("user_followers", IntegerType, true),
      StructField("user_friends", IntegerType, true)
    ))

    // 4. Convert RDD to DataFrame
    val tweetsDF = spark.createDataFrame(tweetsRDD, schema)

    // 5. Write to CSV
    tweetsDF.write.mode("overwrite").option("header", "true").csv("output_tweets_features")

    // 6. Stop session
    spark.stop()
  }
}
