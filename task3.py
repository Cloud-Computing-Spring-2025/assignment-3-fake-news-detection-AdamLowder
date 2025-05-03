from pyspark.sql import SparkSession
from pyspark.sql.functions import concat_ws, col, split, udf
from pyspark.ml.feature import HashingTF, IDF, StringIndexer
from pyspark.ml.linalg import Vectors
from pyspark.sql.types import StringType

# Initialize spark session
spark = SparkSession.builder.appName("FakeNewsClassification").getOrCreate()

# Load the cleaned/tokenized csv from task 2
df = spark.read.csv("task2_output.csv", header=True, inferSchema=True)

# Split the 'filtered_words' back to array (ensure no extra characters)
df = df.withColumn("filtered_words", split(col("filtered_words"), ",\s*"))

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
    # Sparse vector format is like (size, [index1, index2, ...], [value1, value2, ...])
    return str(vector)

# Register the UDF
vector_to_str_udf = udf(vector_to_str, StringType())

# Apply the UDF to convert the 'features' column to a string format
final_df = indexed_df.select("id", "filtered_words", "features", "label_index")

# Convert the vector column to string
final_df = final_df.withColumn("features", vector_to_str_udf(col("features")))

# To ensure that the 'filtered_words' column matches the expected format (cleaned list of words)
final_df = final_df.withColumn("filtered_words", concat_ws(",", col("filtered_words")))

# Save to CSV
final_df.toPandas().to_csv("task3_output.csv", index=False)
print("Task 3 output saved as task3_output.csv")