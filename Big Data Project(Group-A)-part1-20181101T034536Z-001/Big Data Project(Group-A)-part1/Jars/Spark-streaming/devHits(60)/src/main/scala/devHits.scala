import org.apache.spark._
import org.apache.spark.streaming._
import org.apache.spark.streaming.kafka._

object devHits {

def main(args:Array[String])
{
val context = new SparkConf().setAppName("devHits")
val sc = new StreamingContext(context, Seconds(10))
sc.checkpoint("/tmp")
val kafkastream = KafkaUtils.createStream(sc , "master:2181" , "spark-streaming-consumer-group", Map("test-topic" -> 1))
val lines = kafkastream.map(_._2)
val split = lines.map(_.split("\t"))
val result = split.map(a => (a(8),a(6).toInt)).reduceByKeyAndWindow((x:Int, y:Int)=>x+y,Seconds(60),Seconds(30))
result.print()
sc.start()
sc.awaitTermination()
}
}
