from flask import Flask, render_template, request
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
import pandas as pd
import json
from elasticsearch import Elasticsearch
app = Flask(__name__)

spark = SparkSession.builder.appName("Random Forest").getOrCreate()
loaded_model = RandomForestClassificationModel.load('/home/rihab/Desktop/BigDataProject/model')
es = Elasticsearch([{'host': 'localhost', 'port': 9200, 'scheme': 'http'}])
def is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

@app.route('/', methods=['GET', 'POST'])
def index():
    error_message = None
    result = None

    if request.method == 'POST':
        if 'file' in request.files:
            json_file = request.files['file']
            if json_file.filename != '':
                try:
                    json_data = json.load(json_file)
                    required_fields = ['Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio']
                    if all(field in json_data for field in required_fields):
                        # Construct DataFrame from JSON data
                        data = pd.DataFrame([json_data])  # Use a list to handle single row data
                        spark_df = spark.createDataFrame(data)
                        assembler = VectorAssembler(inputCols=spark_df.columns, outputCol="features")
                        spark_df = assembler.transform(spark_df)
                        predictions = loaded_model.transform(spark_df)
                        result = 'Occupied' if predictions.select('prediction').collect()[0][0] == 1.0 else 'Empty'
                        # Save JSON data to be indexed later
                        data_to_store = {
                            'Temperature': json_data['Temperature'],
                            'Humidity': json_data['Humidity'],
                            'Light': json_data['Light'],
                            'CO2': json_data['CO2'],
                            'HumidityRatio': json_data['HumidityRatio'],
                            'Prediction': result
                        }
                    else:
                        error_message = "Uploaded JSON is missing required fields."
                except json.JSONDecodeError:
                    error_message = "Uploaded file is not a valid JSON."
        else:  # Process form inputs
            temperature = request.form['temperature']
            humidity = request.form['humidity']
            light = request.form['light']
            co2 = request.form['co2']
            humidity_ratio = request.form['humidity_ratio']
            
            if not all(is_float(val) for val in [temperature, humidity, light, co2, humidity_ratio]):
                error_message = "Inputs must be floating-point numbers."
            else:
                temperature = float(temperature)
                humidity = float(humidity)
                light = float(light)
                co2 = float(co2)
                humidity_ratio = float(humidity_ratio)
                
                data = pd.DataFrame({
                    'Temperature': [temperature],
                    'Humidity': [humidity],
                    'Light': [light],
                    'CO2': [co2],
                    'HumidityRatio': [humidity_ratio]
                })
                
                spark_df = spark.createDataFrame(data)
                assembler = VectorAssembler(inputCols=spark_df.columns, outputCol="features")
                spark_df = assembler.transform(spark_df)
                predictions = loaded_model.transform(spark_df)
                result = 'Occupied' if predictions.select('prediction').collect()[0][0] == 1.0 else 'Empty'
                if result:
                    data_to_store = {
                    'Temperature': temperature,
                    'Humidity': humidity,
                    'Light': light,
                    'CO2': co2,
                    'HumidityRatio': humidity_ratio,
                    'Prediction': result
                }



        if data_to_store:  # Index data into Elasticsearch if there's something to store
            es.index(index='occupancy_predictions', body=data_to_store)
    return render_template('index.html', result=result, error=error_message)

if __name__ == '__main__':
    app.run(debug=True)
