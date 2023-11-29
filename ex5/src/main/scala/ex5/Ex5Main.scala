package ex5

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import scala.annotation.tailrec


object Ex5Main extends App {
	val spark = SparkSession.builder()
                          .appName("ex5")
                          .config("spark.driver.host", "localhost")
                          .master("local")
                          .getOrCreate()

  // suppress informational log messages related to the inner working of Spark
  val sc: SparkContext = spark.sparkContext
  sc.setLogLevel("WARN")

  // There are three scientific articles in the directory src/main/resources/articles/
  // The call sc.textFile(...) returns an RDD consisting of the lines of the articles:
  val articlesRdd: RDD[String] = sc.textFile("src/main/resources/articles/*")


  printTaskLine(1)

  // Task #1: How do you get the first 10 lines as an Array?
  val lines10: Array[String] = articlesRdd.take(10)
  lines10.foreach(println)



  printTaskLine(2)
  // Task #2: Compute how many lines there are in total in the articles.
  //          And then count the total number of words in the articles
  //          You can assume that words in each line are separated by the space character (i.e. " ")
  val nbrOfLines: Long = articlesRdd.count()
  println(s"#lines = ${nbrOfLines}")
  println(articlesRdd.flatMap(line => line.split(" ")).count())
  val words = articlesRdd.flatMap(line => line.split(" ")).map(w => (w, 1)).reduce((a,b)=> (a._1,a._2+b._2))._2


  println(s"#words = ${words}")

  printTaskLine(3)
  // Task #3: What is the count of non-white space characters? (it is enough to count the non " "-characters for this)
  //          And how many numerical characters are there in total? (i.e., 0, 1, 2, ..., 9 characters)
  val chars: Long = articlesRdd.flatMap(line => line.filter(c => !c.isSpaceChar)).count()
  println(s"#chars = ${chars}")

  val numChars: Long = articlesRdd.flatMap(line => line.filter(c => c.isDigit)).count()
  println(s"#numChars = ${numChars}")

  printTaskLine(4)
  // Task #4: How many 5-character words that are not "DisCo" are there in the corpus?
  //          And what is the most often appearing 5-character word (that is not "DisCo") and how many times does it appear?
  val wordsWith5Letters = articlesRdd.flatMap(line => line.split(" ")).filter(w => w.length == 5 &&
    !w.equals("DisCo") && w.charAt(0).isLetter && w.charAt(4).isLetter)
  val words5Count: Long = wordsWith5Letters.count()
  println(s"5-character words: ${words5Count}")

   val word5Counts = wordsWith5Letters.map(w => (w.toLowerCase(), 1))
     .reduceByKey(_+_).sortBy(_._2, false)
  val commonWord: String = word5Counts.take(1).head._1
  val commonWordCount: Int = word5Counts.take(1).head._2
  println(s"The most common word is '${commonWord}' and it appears ${commonWordCount} times")

  // You are given a factorization function that returns the prime factors for a given number:
  // For example, factorization(28) would return List(2, 2, 7)
  def factorization(number: Int): List[Int] = {
    @tailrec
    def checkFactor(currentNumber: Int, factor: Int, factorList: List[Int]): List[Int] = {
      if (currentNumber == 1) factorList
      else if (factor * factor > currentNumber) factorList :+ currentNumber
      else if (currentNumber % factor == 0) checkFactor(currentNumber / factor, factor, factorList :+ factor)
      else checkFactor(currentNumber, factor + 1, factorList)
    }

    if (number < 2) List(1)
    else checkFactor(number, 2, List.empty)
  }

  printTaskLine(5)
  // Task #5: You are given a sequence of integers and a factorization function.
  //          Using them create a pair RDD that contains the integers and their prime factors.
  //          Get all the distinct prime factors from the RDD.
  val values: Seq[Int] = 12.to(17) ++ 123.to(127) ++ 1234.to(1237)

  val factorRdd: RDD[(Int, List[Int])] = spark.sparkContext.parallelize(values.map(v => (v, factorization(v))))
  factorRdd.collect().foreach({ case (n, factors) => println(s"$n: ${factors.mkString(",")}") })


  val distinctPrimes: List[Int] = factorRdd.values.flatMap(v => v).distinct().collect().toList.sorted
  println(s"distinct primes: ${distinctPrimes.mkString(", ")}")

  printTaskLine(6)
  // Task #6: Here is a code snippet. Explain how it works.

  //The lines are taken first, then each word is taken into an array,
  //then each word is mapped, the word is the key and each word is given value 1,
  //then it counts how many time each key appears in the lyrics,
  //in other words, it counts the frequency of each word in the lyrics using map and reduce functions
  val lyricsRdd = sc.textFile("lyrics/*.txt")

  val lyricsCount = lyricsRdd.flatMap(line => line.split(" "))
    .map(word => (word, 1))
    .reduceByKey((v1, v2) => v1 + v2)

  lyricsCount.collect().foreach(println)


  def printTaskLine(taskNumber: Int): Unit = {
    println(s"======\nTask $taskNumber\n======")
  }
}
