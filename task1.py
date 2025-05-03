from pyspark.sql import SparkSession

#Initialize spark session
spark = SparkSession.builder.appName("FakeNewsClassification").getOrCreate()
#Load csv with inferred schema
df = spark.read.csv("fake_news_sample.csv", header=True, inferSchema=True)
#Create a temporary view
df.createOrReplaceTempView("news_data")

#Show first 5 rows
print("=== First 5 Rows ===")
df.show(5)
#Count total number of articles
total_articles = spark.sql("SELECT COUNT(*) AS total FROM news_data")
print("=== Total Number of Articles ===")
total_articles.show()
#Retrieve distinct labels
distinct_labels = spark.sql("SELECT DISTINCT label FROM news_data")
print("=== Distinct Labels ===")
distinct_labels.show()

#Save the dataframe to csv
#save first 5 rows as csv
df.limit(5).toPandas().to_csv("task1_output.csv", index=False)
print("Task 1 output saved as task1_output.csv")