from flask import Flask, render_template, request
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load the saved model
spark = SparkSession.builder.appName("Random Forest").getOrCreate()
loaded_model = RandomForestClassificationModel.load('/home/rihab/Desktop/BigDataProject/model')

# Define a route for the home page
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Retrieve inputs from the form
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        light = float(request.form['light'])
        co2 = float(request.form['co2'])
        humidity_ratio = float(request.form['humidity_ratio'])
        
        # Create a DataFrame from the inputs
        data = pd.DataFrame({
            'Temperature': [temperature],
            'Humidity': [humidity],
            'Light': [light],
            'CO2': [co2],
            'HumidityRatio': [humidity_ratio]
        })
        
        # Convert Pandas DataFrame to Spark DataFrame
        spark_df = spark.createDataFrame(data)
        
        # Apply VectorAssembler transformation
        assembler = VectorAssembler(inputCols=spark_df.columns, outputCol="features")
        spark_df = assembler.transform(spark_df)
        
        # Make predictions
        predictions = loaded_model.transform(spark_df)
        result = predictions.select('prediction').collect()[0][0]
        
        return render_template('index.html', result=result)
    
    return render_template('index.html', result=None)

if __name__ == '__main__':
    app.run(debug=True)
