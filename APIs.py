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
from pyspark.sql.functions import lit
import random


# Assuming `data` is a dictionary containing the input data

# Create a SparkSession
spark = SparkSession.builder \
    .appName("Example") \
    .getOrCreate()

# Convert the dictionary to a DataFrame


# Transform the data using the loaded pipeline


# Convert [10] to a vector
vector = Vectors.dense([10])


PRICES_MODEL_PATH = "linear_regression_model"
DELAYS_MODEL_PATH = "delays_lr_model"

PRICES_PIPELINE_PATH = "pipeline_model2"
DELAYS_PIPELINE_PATH = "delays_pipeline"


CARRIER_DELAY = {'UA': 3.423222419985118,
 'NK': 3.2118444457475714,
 'AA': 4.7917388244190064,
 'EV': 5.342242695384107,
 'B6': 7.299943174643263,
 'PT': 2.539269115342735,
 'DL': 3.7599815035051902,
 'OO': 6.209415580208714,
 'F9': 5.000508332704653,
 'YV': 5.67237268244944,
 'MQ': 2.6363881908638844,
 'C5': 6.434559493742045,
 'OH': 3.49215333495421,
 'EM': 2.1525951715607676,
 'HA': 3.0608543863527027,
 'G4': 5.794402889991456,
 'ZW': 4.268656961022644,
 'YX': 3.003557364412194,
 'AS': 2.5637682170002094,
 'QX': 2.2097902553837927,
 'G7': 4.747592106149391,
 'WN': 3.1851960736430476,
 '9E': 3.1854992326419103,
 'CP': 4.858521951123864,
 'AX': 6.546503974809003,
 'KS': 8.625425170068027,
 '9K': 2.388185654008439,
 'VX': 3.0799443058536866
}


LATE_AIRCRAFT_DELAY = {'UA': 5.1743180972118035,
 'NK': 3.694704255360738,
 'AA': 4.874551629303676,
 'EV': 5.129514361667639,
 'B6': 7.2672119548665535,
 'PT': 4.982086420736728,
 'DL': 2.304132494997973,
 'OO': 4.003186489827333,
 'F9': 7.564506162860458,
 'YV': 6.064572802154132,
 'MQ': 4.095667314583926,
 'C5': 10.02854928666937,
 'OH': 5.852045276494583,
 'EM': 6.494479820422374,
 'HA': 1.3495462169577033,
 'G4': 7.319008289238979,
 'ZW': 7.082420583977493,
 'YX': 4.000902014364639,
 'AS': 2.7092320461748516,
 'QX': 2.5150772280068066,
 'G7': 7.782649693540782,
 'WN': 4.315058973990045,
 '9E': 3.5869364059012443,
 'CP': 6.185060033072111,
 'AX': 10.59635040264299,
 'KS': 6.386479591836735,
 '9K': 1.5123568414707655,
 'VX': 2.859430295295005}


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
        loaded_pipeline = PipelineModel.load(PRICES_PIPELINE_PATH)
    except Exception as e:
        print(e, "load failed")
    # Load the trained model

    try:
        loaded_model = LinearRegressionModel.load(PRICES_MODEL_PATH)
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
    

    # Add a new column "totalTravelDistance" with a constant value of 100
    # transformed_data = transformed_data.withColumn("totalTravelDistance", lit(1844))

    # # Add another new column "travelDurationMinutes" with a constant value of 100
    # transformed_data = transformed_data.withColumn("travelDurationMinutes", lit(740))

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
    price_prediction = [row.prediction for row in results][0]
    # @karthik /35.0 is us rigging the system
    
    return price_prediction


def predict_delays(data):
    try:
        loaded_pipeline_2 = PipelineModel.load(DELAYS_PIPELINE_PATH)
    except Exception as e:
        print(e, "load failed")
    # Load the trained model

    try:
        loaded_model_2 = LinearRegressionModel.load(DELAYS_MODEL_PATH)
        a = vector

    except Exception as e:
        print(e, "model failed")

    
    data_2 = {
        "FlightWeek": 17,
        "Origin": "LAS",
        "Dest":"SLC",
        "arrDaypart":"night",
        "depDaypart": "night",
        "CarrierDelay":0.0,          # @karthik -> You can use CARRIER_DELAY["DL"] (DL is Operating_Airline)
        "WeatherDelay":0.0,          # @karthik -> You can use WEATHER_DELAY["Origin_Dest"] (This weather delay needs to be imported from weatherDelay.json)
        "NASDelay":0.0,              # @karthik -> You can use NAS_DELAY["Origin_Dest"] (This NAS delay needs to be imported from nasDelay.json)
        "SecurityDelay":0.0,         # @karthik -> You can use SECURITY_DELAY["Origin"] (This Security delay needs to be imported from securityDelay.json)   
        "LateAircraftDelay":0.0,     # @karthik -> You can use LATE_AIRCRAFT_DELAY["DL"] (DL is Operating_Airline)
        "Distance":368.0,
        "Operating_Airline": "DL"
    }





    try:
        data_df_2 = spark.createDataFrame(pd.DataFrame([data_2]))
        # Transform input data using the loaded pipeline
        transformed_data_2 = loaded_pipeline_2.transform(data_df_2)
        print(transformed_data_2.toPandas().head())
    except Exception as e:
        print(e, "transform failed")

    # Make predictions using the loaded model

    columns_to_drop = [
        "FlightWeek",
        "Origin",
        "Dest",
        "arrDaypart",
        "depDaypart",
        "Operating_Airline"
    ]

    # Drop the specified columns
    transformed_data_2 = transformed_data_2.drop(*columns_to_drop)
    

    # Add a new column "totalTravelDistance" with a constant value of 100
    # transformed_data = transformed_data.withColumn("totalTravelDistance", lit(1844))

    # # Add another new column "travelDurationMinutes" with a constant value of 100
    # transformed_data = transformed_data.withColumn("travelDurationMinutes", lit(740))

    testAssembler_2 = VectorAssembler(inputCols=transformed_data_2.columns,
                                    outputCol="features")
    testOutput_2 = testAssembler_2.transform(transformed_data_2)

    print(testOutput_2.printSchema())
    predictions_2 = loaded_model_2.transform(testOutput_2)
    print(predictions_2.printSchema())
    print(predictions_2.select("prediction").limit(1).toPandas().head())
    
    # Extract prediction results
    try:
        results_2 = predictions_2.select('prediction').collect()
    except Exception as e:
        print(e, "prediation failed")
    # Convert results to JSON
    delay_prediction_2 = [row.prediction for row in results_2][0]
    # @karthik /25.0 is us rigging the system
    res = (delay_prediction_2 % 25)  * random.uniform(-1, 1)
    return res
# 480.995284





@app.route('/getCityNames', methods=["GET"])
def getCityNames():
    # Load the pipeline
    print("something")
    predict_delays({"test":"test"})
    print("\n\n\n\n\n\n\n\n\another osmething")
    predict_prices({"test":"test"})
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
