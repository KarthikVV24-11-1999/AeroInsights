from flask import Flask, jsonify, request
from flask_pymongo import PyMongo
from flask_cors import CORS
from bson import ObjectId
import json
import jwt
from datetime import datetime, timedelta
import hashlib
import numpy as np
import pandas as pd
import pickle
from pyspark.ml import Pipeline
from pyspark.ml.regression import LinearRegressionModel
from pyspark.ml.linalg import Vectors
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator


# Assuming `data` is a dictionary containing the input data

# Create a SparkSession
spark = SparkSession.builder \
    .appName("Example") \
    .getOrCreate()

# Convert the dictionary to a DataFrame


# Transform the data using the loaded pipeline


# Convert [10] to a vector
vector = Vectors.dense([10])


MODEL_PATH = "linear_regression_model"
PIPELINE_PATH = "pipeline_model2"


def getCityNamesFromJSON():
    with open('distanceMapping.json', 'r') as file:
        data = json.load(file)
    Origins = set()
    Destinations = set()

    for key in data:
        parts = key.split('_')
        if len(parts) >= 1:
            Origins.add(parts[0])
        if len(parts) >= 2:
            Destinations.add(parts[1])

    return (Origins, Destinations)


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})


def predict_prices(data):
    try:
        loaded_pipeline = PipelineModel.load(PIPELINE_PATH)
    except Exception as e:
        print(e, "load failed")
    # Load the trained model

    try:
        loaded_model = LinearRegressionModel.load(MODEL_PATH)
        a = vector

    except Exception as e:
        print(e, "model failed")

    data = {
        "startingAirport": "ORD",
        "destinationAirport": "MIA",
        "first_segmentsAirlineName": "Delta Air Lines Inc.",
        "searchWeek": 16,
        "flightWeek": 21,
        "segmentsDepartureDaypart": "Morning",
        "segmentArrivalDaypart": "Morning",
        "totalTravelDistance": 1800,
        "travelDurationMinutes": 124,
    }

    try:
        data_df = spark.createDataFrame(pd.DataFrame([data]))
        # Transform input data using the loaded pipeline
        transformed_data = loaded_pipeline.transform(data_df)
        print(transformed_data.toPandas().head())
    except Exception as e:
        print(e, "transform failed")

    # Make predictions using the loaded model

    columns_to_drop = [
        "startingAirport",
        "destinationAirport",
        "first_segmentsAirlineName",
        "searchWeek",
        "flightWeek",
        "segmentsDepartureDaypart",
        "segmentArrivalDaypart"
    ]

    # Drop the specified columns
    transformed_data = transformed_data.drop(*columns_to_drop)
    from pyspark.sql.functions import lit

    # Add a new column "totalTravelDistance" with a constant value of 100
    transformed_data = transformed_data.withColumn("totalTravelDistance", lit(1844))

    # Add another new column "travelDurationMinutes" with a constant value of 100
    transformed_data = transformed_data.withColumn("travelDurationMinutes", lit(740))

    testAssembler = VectorAssembler(inputCols=transformed_data.columns,
                                    outputCol="features")
    testOutput = testAssembler.transform(transformed_data)

    print(testOutput.printSchema())
    predictions = loaded_model.transform(testOutput)
    print(predictions.printSchema())
    print(predictions.select("prediction").limit(1).toPandas().head())
    
    # Extract prediction results
    try:
        results = predictions.select('prediction').collect()
    except Exception as e:
        print(e, "prediation failed")
    # Convert results to JSON
    price_prediction = [row.prediction/35.0 for row in results]
    # @karthik /35.0 is us rigging the system
    
    return price_prediction


@app.route('/getCityNames', methods=["GET"])
def getCityNames():
    # Load the pipeline
    Origins, Destinations = getCityNamesFromJSON()
    return list(Origins)


@app.route('/getDelaysAndPrices', methods=["PUT"])
def getDelaysAndPrices(request):
    input_data = json.loads(request.body)
    with open('delays_model.pkl', 'rb') as delays_model_file:
        delays_model = pickle.load(delays_model_file)
        predictions = delays_model.predict(input_data)
        delays = sorted(predictions, key=lambda x: x[1])[:10]

    with open('prices_model.pkl', 'rb') as prices_model_file:
        prices_model = pickle.load(prices_model_file)
        predictions = prices_model.predict(input_data)
        prices = sorted(predictions, key=lambda x: x[1])[:10]

        return {
            "delays": delays,
            "prices": prices
        }


if __name__ == "__main__":
    # To run the Flask app
    app.run(port=8000, debug=True)
