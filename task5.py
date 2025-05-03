from pyspark.sql import SparkSession
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import pandas as pd

#Initialize spark session
spark = SparkSession.builder.appName("FakeNewsClassification").getOrCreate()

#Load predictions from task 4
predictions = spark.read.csv("task4_output.csv", header=True, inferSchema=True)
#Cast prediction and label_index columns to double (needed for evaluator)
predictions = predictions.withColumn("label_index", predictions["label_index"].cast("double"))
predictions = predictions.withColumn("prediction", predictions["prediction"].cast("double"))
#Initialize evaluators
accuracy_evaluator = MulticlassClassificationEvaluator(labelCol="label_index", predictionCol="prediction", metricName="accuracy")
f1_evaluator = MulticlassClassificationEvaluator(labelCol="label_index", predictionCol="prediction", metricName="f1")
#Compute metrics
accuracy = accuracy_evaluator.evaluate(predictions)
f1_score = f1_evaluator.evaluate(predictions)

#Print in markdown table
print("| Metric    | Value  |")
print("|-----------|--------|")
print(f"| Accuracy  | {accuracy:.2f} |")
print(f"| F1 Score | {f1_score:.2f} |")

#Save metrics to csv
metrics_df = pd.DataFrame({"Metric": ["Accuracy", "F1 Score"], "Value": [round(accuracy, 2), round(f1_score, 2)]})
metrics_df.to_csv("task5_output.csv", index=False)
print("Task 5 output saved as task5_output.csv")