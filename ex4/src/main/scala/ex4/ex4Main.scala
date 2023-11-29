package ex4

import org.apache.spark.SparkConf

import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.regression.LinearRegressionModel

import org.apache.spark.rdd.RDD

import org.apache.spark.sql.Column
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.Row
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.SQLContext

import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.types.{ArrayType, StringType, StructField, IntegerType, DoubleType}
import org.apache.spark.sql.types.DataType
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.functions.{count, sum, min, max, asc, desc, udf, to_date, avg}

import org.apache.spark.storage.StorageLevel

import scala.language.postfixOps

import org.apache.log4j.Logger
import org.apache.log4j.Level


object Ex4Main extends App {
	val spark = SparkSession.builder()
                          .appName("ex4")
                          .config("spark.driver.host", "localhost")
                          .master("local")
                          .getOrCreate()

  // suppress informational or warning log messages related to the inner working of Spark
  spark.sparkContext.setLogLevel("ERROR")


  // Wikipedia defines: Simple Linear Regression
  //
  // In statistics, simple linear regression is a linear regression model with a single explanatory variable.
  // That is, it concerns two-dimensional sample points with one independent variable and one dependent variable
  // (conventionally, the x and y coordinates in a Cartesian coordinate system) and finds a linear function (a non-vertical straight line)
  // that, as accurately as possible, predicts the dependent variable values as a function of the independent variables. The adjective simple
  // refers to the fact that the outcome variable is related to a single predictor.

  // You are given an dataRDD of Rows (first element is x and the other y). We are aiming at finding simple linear regression model
  // for the dataset using MLlib. I.e. find function f so that y ~ f(x)

  val hugeSequenceOfXYData = Seq(
    Row(0.0, 0.0), Row(0.3, 0.5), Row(0.9, 0.8), Row(1.0, 0.8),
    Row(2.0, 2.2), Row(2.2, 2.4), Row(3.0, 3.7), Row(4.0, 4.3),
    Row(1.5, 1.4), Row(3.2, 3.9), Row(3.5, 4.1), Row(1.2, 1.1)
  )
  val dataRDD: RDD[Row] = spark.sparkContext.parallelize(hugeSequenceOfXYData)



  printTaskLine(1)
  // Task 1: Transform dataRDD to a DataFrame dataDF, with two columns "X" (of type Double) and "label" (of type Double).
  //         (The default dependent variable name is "label" in MLlib)
  val schema = new StructType(Array(new StructField("X", DoubleType, true), new StructField("label", DoubleType, true)))
  val dataDF: DataFrame = spark.createDataFrame(dataRDD, schema)
  println("dataDF")
  dataDF.show()


  // Let's split the data into training and testing datasets
  val trainTest: Array[DataFrame] = dataDF.randomSplit(Array(0.7, 0.3))
  val trainingDF: DataFrame = trainTest(0)
  println("trainingDF")
  trainingDF.show()


  printTaskLine(2)
  // Task 2: Create a VectorAssembler for mapping input column "X" to "features" column and
  //         apply it to trainingDF in order to create assembled training data frame
  val vectorAssembler: VectorAssembler = new VectorAssembler()
    .setInputCols(Array("X"))
    .setOutputCol("features")


  val assembledTrainingDF: DataFrame = vectorAssembler.transform(trainingDF)
  assembledTrainingDF.show()

  printTaskLine(3)
  // Task 3: Create a LinearRegression object and fit using the training data to get a LinearRegressionModel object
  val lr: LinearRegression = new LinearRegression()

  println(lr.explainParams())

  val lrModel: LinearRegressionModel = lr.fit(assembledTrainingDF)
  lrModel.summary.predictions.show()

  printTaskLine(4)
  // Task 4: Apply the model to the whole dataDF

  val allPredictions: DataFrame = lrModel.transform(vectorAssembler.transform(dataDF))
  allPredictions.show()

  printTaskLine(5)
  // Task 5: Use the LinearRegressionModel to predict y for values [-0.5, 3.14, 7.5]

  val df = spark.createDataFrame(spark.sparkContext.parallelize(Seq(Row(-0.5, 0.0), Row(3.14, 0.0), Row(7.5, 0.0))), schema)
  lrModel.transform(vectorAssembler.transform(df)).show()

  printTaskLine(6)
  // Task 6: File "data/numbers.csv" contains one column "X" with several more x values.
  //         Use the LinearRegressionModel to predict the corresponding y values for them.
  val numbers = spark
    .read
    .option("inferSchema"
      ,
      "true")
    .option("header"
      ,
      "true")
    .csv("data/numbers.csv")

  val numberPredictionsDF: DataFrame = lrModel.transform(vectorAssembler.transform(numbers))
  numberPredictionsDF.printSchema()
  numberPredictionsDF.show()

  printTaskLine(7)
  // Task 7: Store the resulting DataFrame from task 6 into the folder "results" in CSV format.
  //         NOTE: It is ok if you get multiple files with long file names
  numberPredictionsDF.drop("features").write.option("header", true).csv("results/predictions")





// Stop the Spark session
  spark.stop()

  def printTaskLine(taskNumber: Int): Unit = {
    println(s"======\nTask $taskNumber\n======")
  }
}
