from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.sql.functions import col, udf
from pyspark.ml.linalg import Vectors
from pyspark.sql.types import ArrayType, FloatType
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Initialize Spark session
spark = SparkSession.builder.appName("Task4_LogisticRegression").getOrCreate()

# Load data from task3_output.csv
data = spark.read.csv("task3_output.csv", header=True, inferSchema=True)

# Define a UDF to convert the string representation of a vector into a sparse vector
def parse_features(features_str):
    # Extract indices and values from the string and return a sparse vector
    indices = [int(i.split(":")[0]) for i in features_str.split(",")]
    values = [float(i.split(":")[1]) for i in features_str.split(",")]
    return Vectors.sparse(len(indices), indices, values)

# Register the UDF
parse_features_udf = udf(parse_features, ArrayType(FloatType()))

# Apply the UDF to convert the features column into a proper vector column
data = data.withColumn("features", parse_features_udf(col("features")))

# Preprocess the data (VectorAssembler to combine feature columns into a single features column)
assembler = VectorAssembler(inputCols=["features"], outputCol="assembled_features")

# Split the data into 80% training and 20% test sets
train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)

# Initialize Logistic Regression Model
lr = LogisticRegression(featuresCol="assembled_features", labelCol="label_index")

# Build the pipeline
pipeline = Pipeline(stages=[assembler, lr])

# Train the model using the training data
model = pipeline.fit(train_data)

# Make predictions on the test data
predictions = model.transform(test_data)

# Select the necessary columns: id, title, label_index, prediction
output = predictions.select("id", "title", "label_index", "prediction")

# Write the predictions to task4_output.csv
output.write.csv("task4_output.csv", header=True)

# Stop the Spark session
spark.stop()