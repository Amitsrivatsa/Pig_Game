import org.apache.spark._
import org.apache.spark.streaming._
object Time_min {
def main(args:Array[String]) {
val conf = new SparkConf().setAppName("TimeUserMin")
val sc = new SparkContext(conf)
val inp = sc.textFile("file:///home/vagrant/goShopping_WebClicks.dat",2)
val split = inp.map(_.split("\t"))
val line = split.map(x => (x(4),x(6).toInt)).reduceByKey(_+_)
val result = line.sortBy(_._2)
result.take(1).foreach(println)
}
}
