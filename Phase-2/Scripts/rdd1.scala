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
