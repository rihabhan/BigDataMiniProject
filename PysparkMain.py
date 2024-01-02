#Necessary libraries.
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
# Create a Spark session on local machine with as many threads as available cores on the machine for processing.

spark = SparkSession.builder.appName("Random Forest").master("local[*]").getOrCreate()

# Read the CSV file into a Spark DataFrame.
#When header='True', Spark will consider the first row of the CSV file as column names and not as data.
#When inferSchema='True', Spark will attempt to automatically infer the data types for each column in the DataFrame based on the contents of the CSV file.

df = spark.read.options(header='True', inferSchema='True', delimiter=',').csv("hdfs://localhost:9000/user/Occupancy.csv")

#Exclude columns 'date' and 'Occupancy' from the list of all columns in the DataFrame into the new list feature_cols.

feature_cols = [col for col in df.columns if col not in ['date', 'Occupancy']]

#Combine multiple feature columns into a single vector column called features

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

# Apply VectorAssembler transformation to df, creating a new DataFrame with an additional column containing the assembled features.

df = assembler.transform(df)

#Seed: To ensure that the same random split will occur each time this code is executed.

(train_data, test_data) = df.randomSplit([0.66, 0.33], seed=42)

# Create the Random Forest Classifier
#labelCol Specifies the name of the column in your dataset that contains the labels or the target variable that the model aims to predict. 
#numTrees Defines the number of decision trees to be included in the random forest. 
#featureSubsetStrategy="auto": algo automatically determines the strategy for selecting the number of features to consider for splitting at each tree node. 
rf_classifier = RandomForestClassifier(labelCol="Occupancy", numTrees=100, featureSubsetStrategy="auto")

# Train the model

model = rf_classifier.fit(train_data)

# Make predictions

predictions = model.transform(test_data)

# Evaluate the model
evaluator = MulticlassClassificationEvaluator(labelCol="Occupancy", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print(f'Accuracy: {accuracy}')
#save the model 
model.write().overwrite().save('hdfs://localhost:9000/user/')
# Stop the Spark session
spark.stop()