name := "kafka-directStream-word-count-1"
version := "1.0"

// https://mvnrepository.com/artifact/org.apache.spark/spark-core_2.11
libraryDependencies += "org.apache.spark" % "spark-core_2.11" % "2.1.1" % "provided"

// https://mvnrepository.com/artifact/org.apache.spark/spark-streaming_2.11
libraryDependencies += "org.apache.spark" % "spark-streaming_2.11" % "2.1.1" % "provided"

// https://mvnrepository.com/artifact/org.apache.spark/spark-streaming-kafka-0-8-assembly_2.11
libraryDependencies += "org.apache.spark" % "spark-streaming-kafka-0-8-assembly_2.11" % "2.1.1"
