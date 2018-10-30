import org.apache.spark._
import org.apache.spark.streaming._
import org.apache.spark.streaming.kafka._

object devHits {

def updatefn(value:Seq[Int],newValue:Option[Int]):Option[Int] =
{
val count = newValue.getOrElse(0) + value.sum
Some(count)
}

def main(args:Array[String])
{
val context = new SparkConf().setAppName("devHits")
val sc = new StreamingContext(context, Seconds(30))
val kafkastream = KafkaUtils.createStream(sc , "master:2181" , "spark-streaming-consumer-group", Map("test-topic" -> 1))
sc.checkpoint("/tmp")
val lines = kafkastream.map(_._2)
val split = lines.map(_.split("\t"))
val result = split.map(a => (a(8),a(6).toInt))
val result1 = result.updateStateByKey[Int](updatefn _)
result1.print()
sc.start()
sc.awaitTermination()
}
}
