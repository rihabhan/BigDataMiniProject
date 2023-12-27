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

# Function to check if a string can be converted to float
def is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

# Define a route for the home page
@app.route('/', methods=['GET', 'POST'])
def index():
    error_message = None  # Initialize error message

    if request.method == 'POST':
        # Retrieve inputs from the form
        temperature = request.form['temperature']
        humidity = request.form['humidity']
        light = request.form['light']
        co2 = request.form['co2']
        humidity_ratio = request.form['humidity_ratio']
        
        # Check if inputs are floats
        if not all(is_float(val) for val in [temperature, humidity, light, co2, humidity_ratio]):
            error_message = "Inputs must be floating-point numbers."
        else:
            # Convert inputs to floats
            temperature = float(temperature)
            humidity = float(humidity)
            light = float(light)
            co2 = float(co2)
            humidity_ratio = float(humidity_ratio)
            
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
    
    return render_template('index.html', result=None, error=error_message)

if __name__ == '__main__':
    app.run(debug=True)
