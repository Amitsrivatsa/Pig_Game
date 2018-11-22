import org.apache.spark.SparkContext._
import org.apache.spark.sql.Row
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types._
import org.apache.spark._
object Test{
case class WebClicksData(searchDate:String,searchTime:String,HostIp:String,cs_method:String,customer_ip:String,domain:String,product:String,productType:String,timeSpent:String,redirectedFrom:String,deviceType:String)
def main(args: Array[String]) {
val spark = SparkSession.builder().appName("Min").config("spark.some.config.option", "some-value").getOrCreate()
import spark.implicits._
val IpLookup_DF = spark.sparkContext.textFile("file:///home/vagrant/goShop_Web").map(_.split(",")).map(a => WebClicksData(a(0),a(1),a(2),a(3),a(4),a(5),a(6),a(7),a(8),a(9),a(10))).toDF()
IpLookup_DF.createOrReplaceTempView("WebClicks")
val result = spark.sql("select customer_ip,sum(timeSpent) as MIN_TIME from webclicks group by customer_ip order by min_time limit 1")
result.show()
}
}
