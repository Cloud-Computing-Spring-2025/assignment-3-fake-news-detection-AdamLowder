from pyspark.sql import SparkSession
from pyspark.sql.functions import concat_ws, col, split, udf, regexp_replace
from pyspark.ml.feature import HashingTF, IDF, StringIndexer
from pyspark.sql.types import StringType

# Initialize spark session
spark = SparkSession.builder.appName("FakeNewsClassification").getOrCreate()

# Load the cleaned/tokenized csv from task 2
df = spark.read.csv("task2_output.csv", header=True, inferSchema=True)

# Split the 'filtered_words' back to array (removing brackets, quotes, and spaces if needed)
df = df.withColumn("filtered_words", regexp_replace("filtered_words", r"[\[\]'\" ]", ""))
df = df.withColumn("filtered_words", split(col("filtered_words"), ","))

# Apply HashingTF
hashing_tf = HashingTF(inputCol="filtered_words", outputCol="raw_features", numFeatures=1000)
featurized_df = hashing_tf.transform(df)

# Apply IDF
idf = IDF(inputCol="raw_features", outputCol="features")
idf_model = idf.fit(featurized_df)
rescaled_df = idf_model.transform(featurized_df)

# Index labels
indexer = StringIndexer(inputCol="label", outputCol="label_index")
indexed_df = indexer.fit(rescaled_df).transform(rescaled_df)

# Function to convert SparseVector to string format
def vector_to_str(vector):
    indices = vector.indices
    values = vector.values
    return ",".join(f"{i}:{v:.6f}" for i, v in zip(indices, values))

# Register the UDF
vector_to_str_udf = udf(vector_to_str, StringType())

# Convert 'features' to readable string and 'filtered_words' back to CSV-friendly string
final_df = indexed_df.withColumn("features", vector_to_str_udf(col("features")))
final_df = final_df.withColumn("filtered_words", concat_ws(",", col("filtered_words")))

# Select required columns
final_df = final_df.select("id", "filtered_words", "features", "label_index")

# Save to CSV using Spark (avoids .toPandas() memory issue)
final_df.coalesce(1).write.csv("task33_output.csv", header=True, mode="overwrite")

print("Task 3 output saved as task33_output.csv")