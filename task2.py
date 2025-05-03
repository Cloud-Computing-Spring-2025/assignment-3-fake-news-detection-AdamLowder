import string

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, udf
from pyspark.ml.feature import Tokenizer, StopWordsRemover
from pyspark.sql.types import ArrayType, StringType

# Initialize Spark session
spark = SparkSession.builder.appName("FakeNewsClassification").getOrCreate()

# Load the csv
df = spark.read.csv("fake_news_sample.csv", header=True, inferSchema=True)

# Convert 'text' column to lowercase
df = df.withColumn("text_lower", lower(col("text")))

# Tokenize text
tokenizer = Tokenizer(inputCol="text_lower", outputCol="words")
tokenized_df = tokenizer.transform(df)

# Remove stopwords
remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
cleaned_df = remover.transform(tokenized_df)

# Function to remove punctuation from words
def remove_punctuation(words):
    return [word.strip(string.punctuation) for word in words]

# Register the UDF
remove_punctuation_udf = udf(remove_punctuation, ArrayType(StringType()))

# Apply the UDF to remove punctuation
cleaned_df = cleaned_df.withColumn("filtered_words", remove_punctuation_udf(col("filtered_words")))

# Select needed columns
result_df = cleaned_df.select("id", "title", "filtered_words", "label")

# Create a temporary view (optional)
result_df.createOrReplaceTempView("cleaned_news")

# Write the tokenized and cleaned output to CSV (keep array as-is)
result_df.toPandas().to_csv("task2_output.csv", index=False)

print("Task 2 output saved as task2_output.csv")