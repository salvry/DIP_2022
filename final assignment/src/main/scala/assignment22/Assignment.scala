package assignment22



import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.col
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.feature.MinMaxScaler
import org.apache.spark.sql.types.{DoubleType, StringType, StructField, StructType}
import org.apache.spark.ml.evaluation.ClusteringEvaluator
import org.apache.spark.storage.StorageLevel

import scala.collection.mutable.ArrayBuffer



class Assignment {

  val spark: SparkSession = SparkSession.builder()
    .appName("ex3")
    .config("spark.driver.host", "localhost")
    .master("local")
    .getOrCreate()

  val schema1 = new StructType(Array(new StructField("a", DoubleType, true),
    new StructField("b", DoubleType, true), new StructField("LABEL", StringType, true)))
  // the data frame to be used in tasks 1 and 4
  val dataD2: DataFrame = spark
    .read
    .schema(schema1)
    .option("header"
      ,
      "true")
    .option("delimiter", ",")
    .csv("data/dataD2.csv")
    .persist(StorageLevel.MEMORY_ONLY)


  val schema2 = new StructType(Array(new StructField("a", DoubleType, true),
    new StructField("b", DoubleType, true), new StructField("c", DoubleType, true)))
  // the data frame to be used in task 2
  val dataD3: DataFrame = spark
    .read
    .schema(schema2)
    .option("header"
      ,
      "true")
    .option("delimiter", ",")
    .csv("data/dataD3.csv")
    .persist(StorageLevel.MEMORY_ONLY)


  // the data frame to be used in task 3 (based on dataD2 but containing numeric labels)
  val dataD2WithLabels: DataFrame = new StringIndexer().setInputCol("LABEL")
    .setOutputCol("label_num").fit(dataD2).transform(dataD2)
  dataD2WithLabels.show()


  val scale: MinMaxScaler = new MinMaxScaler().setInputCol("features").setOutputCol("scaledFeatures")
  val km: KMeans = new KMeans().setFeaturesCol("scaledFeatures").setSeed(1L)
  val va = new VectorAssembler().setInputCols(Array("a", "b"))
    .setOutputCol("features")
  val pipeline: Pipeline = new Pipeline().setStages(Array(va, scale))

  def task1(df: DataFrame, k: Int): Array[(Double, Double)] = {

    val transformedData = pipeline.fit(df).transform(df)

    km.setK(k)
    val kmModel = km.fit(transformedData)
    val cc = kmModel.clusterCenters.map(c => (c(0), c(1)))
    cc.foreach(cc => println(cc._1 + ", " + cc._2))
    return cc

  }


  def task2(df: DataFrame, k: Int): Array[(Double, Double, Double)] = {

    va.setInputCols(Array("a", "b", "c"))


    val transformedData = pipeline.fit(df).transform(df)


    km.setK(k)
    val kmModel = km.fit(transformedData)
    val cc = kmModel.clusterCenters.map(c => (c(0), c(1), c(2)))
    cc.foreach(cc => println(cc._1 + ", " + cc._2 + ", " + cc._3))
    return cc
  }

  def task3(df: DataFrame, k: Int): Array[(Double, Double)] = {


    va.setInputCols(Array("a", "b", "label_num"))

    pipeline.setStages(Array(va, scale))

    val transformedData = pipeline.fit(df).transform(df)


    km.setK(k)

    val kmModel = km.fit(transformedData)


    val clustersWithFatal = kmModel.summary.predictions
      .filter(col("LABEL") === "Fatal")
      .groupBy("prediction")
      .count()
      .orderBy(col("count").desc)
      .select("prediction")
      .take(2)


    val index1 = clustersWithFatal(0)(0).toString.toInt
    val index2 = clustersWithFatal(1)(0).toString.toInt


    val cc = kmModel.clusterCenters.map(c => (c(0), c(1)))


    val clusterCentersWithHighestFatal = Array(cc(index1), cc(index2))
    return clusterCentersWithHighestFatal
  }


  // Parameter low is the lowest k and high is the highest one.
  def task4(df: DataFrame, low: Int, high: Int): Array[(Int, Double)] = {

    va.setInputCols(Array("a", "b"))

    pipeline.setStages(Array(va, scale))

    val transformedData = pipeline.fit(df).transform(df)


    val evaluator = new ClusteringEvaluator().setFeaturesCol("scaledFeatures")
    val scores = ArrayBuffer[(Int, Double)]()

    def calcSilhouette(k: Int) = {
      km.setK(k)
      val kmModel4 = km.fit(transformedData)
      val predictions = kmModel4.transform(transformedData)
      val silhouette = evaluator.evaluate(predictions)

      val silhouetteScore: (Int, Double) = (k, silhouette)
      scores += silhouetteScore
    }

    (low to high).toSeq.foreach(k => calcSilhouette(k))


    scores.foreach(s => println(s))

    return scores.toArray

  }
}
