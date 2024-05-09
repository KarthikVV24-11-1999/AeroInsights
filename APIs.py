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
from itertools import groupby
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


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

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

AIRLINES = [
    "American Airlines Inc.", 
    "Delta Air Lines Inc.", 
    "United Air Lines Inc.", 
    "JetBlue Airways", 
    "Alaska Airlines Inc.", 
    "Frontier Airlines Inc.", 
    "Southwest Airlines Co.", 
    "Cape Air", 
    "Peninsula Airways Inc.", 
    "Compass Airlines", 
    "SkyWest Airlines Inc.", 
    "Mesa Airlines Inc.", 
    "Hawaiian Airlines Inc.", 
    "Spirit Airlines Inc.", 
] 

ALL_AIRLINES_WITH_CODES = {
    'Southwest Airlines Co.': 'WN',
    'Delta Air Lines Inc.': 'DL',
    'SkyWest Airlines Inc.': 'OO',
    'American Airlines Inc.': 'AA',
    'United Air Lines Inc.': 'UA',
    'ExpressJet Airlines Inc.': 'EV',
    'JetBlue Airways': 'B6',
    'Allegiant Air': 'G4',
    'GoJet Airlines, LLC d/b/a United Express': 'G7',
    'Endeavor Air Inc.': '9E',
    'Alaska Airlines Inc.': 'AS',
    'Spirit Airlines Inc.': 'NK',
    'Mesa Airlines Inc.': 'YV',
    'Frontier Airlines Inc.': 'F9',
    'Empire Airlines Inc.': 'EM',
    'Republic Airlines': 'YX',
    'Horizon Air': 'QX',
    'Commutair Aka Champlain Enterprises, Inc.': 'C5',
    'Trans States Airlines': 'AX',
    'Hawaiian Airlines Inc.': 'HA',
    'Air Wisconsin Airlines Corp': 'ZW',
    'Virgin America': 'VX',
    'Capital Cargo International': 'PT',
    'Compass Airlines': 'CP',
    'Envoy Air': 'MQ',
    'Comair Inc.': 'OH',
    'Peninsula Airways Inc.': 'KS',
    'Cape Air': '9K',
}

DAYPARTS = [
    "morning", 
    "afternoon", 
    "evening", 
    "night", 
]

CARRIER_DELAY = {
    'UA': 3.423222419985118,
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
    'VX': 3.0799443058536866,
}

LATE_AIRCRAFT_DELAY = {
    'UA': 5.1743180972118035,
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
    'VX': 2.859430295295005, 
}


def getAirportObjectsFromJSON():
    with open('airportCodeToNameMapping.json', 'r') as file:
        data = json.load(file)
    objects = [{'id': key, 'code': key, 'name': value} for key, value in data.items()]
    sorted_objects = sorted(objects, key=lambda x: x['name'])
    return sorted_objects

def buildDataLists(userData):
    data_list = []
    data_2_list = []
    origin, destination, departureDayParts, arrivalDayParts, date, searchDate = userData.values()

    flightWeek = datetime.fromisoformat(date.split('.')[0]).isocalendar()[1]
    searchWeek = datetime.fromisoformat(searchDate.split('.')[0]).isocalendar()[1]
 
    departureDayParts = [dayPart['id'] for dayPart in departureDayParts if dayPart['checked']]
    arrivalDayParts = [dayPart['id'] for dayPart in arrivalDayParts if dayPart['checked']]  

    if departureDayParts == []:
        departureDayParts = DAYPARTS  

    if arrivalDayParts == []:
        arrivalDayParts = DAYPARTS  

    with open('distanceMapping.json', 'r') as file:
        distances = json.load(file)
    orig_dest = origin['name'] + '_' + destination['name']  
    totalTravelDistance = distances.get(orig_dest, 0.0)  

    with open('weatherDelay.json', 'r') as file:
        weatherDelays = json.load(file)
    orig_dest = origin['code'] + '_' + destination['code']
    weatherDelay = weatherDelays.get(orig_dest, 0.0)  

    with open('nasDelay.json', 'r') as file:
        nasDelays = json.load(file)
    orig_dest = origin['code'] + '_' + destination['code']
    nasDelay = nasDelays.get(orig_dest, 0.0)  

    with open('securityDelay.json', 'r') as file:
        securityDelays = json.load(file)
    securityDelay = securityDelays.get(origin['code'], 0.0)

    for airline in AIRLINES:
        for departureDayPart in departureDayParts:  
            for arrivalDayPart in arrivalDayParts:  
                data = {
                    "startingAirport": origin['code'],
                    "destinationAirport": destination['code'],
                    "first_segmentsAirlineName": airline,
                    "searchWeek": searchWeek,
                    "flightWeek": flightWeek,
                    "segmentsDepartureDaypart": departureDayPart,
                    "segmentArrivalDaypart": arrivalDayPart,
                    "totalTravelDistance": totalTravelDistance,
                    "travelDurationMinutes": 124,
                }
                data_list.append(data)

                airline_code = ALL_AIRLINES_WITH_CODES[airline]
                data_2 = {
                    "FlightWeek": flightWeek,
                    "Origin": origin['code'],
                    "Dest": destination['code'],
                    "arrDaypart": arrivalDayPart,
                    "depDaypart": departureDayPart,
                    "CarrierDelay": CARRIER_DELAY[airline_code],
                    "WeatherDelay": weatherDelay,
                    "NASDelay": nasDelay,
                    "SecurityDelay": securityDelay,
                    "LateAircraftDelay": LATE_AIRCRAFT_DELAY[airline_code],
                    "Distance": totalTravelDistance,
                    "Operating_Airline": airline_code,
                }
                data_2_list.append(data_2)

    return {'data_list': data_list, 'data_2_list': data_2_list}


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

    try:
        data_df = spark.createDataFrame(pd.DataFrame(data))
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
    print(predictions.select("prediction", "first_segmentsAirlineName_index").toPandas())
    
    # Extract prediction results
    try:
        results = predictions.select('prediction', 'first_segmentsAirlineName_index').collect()  
        # print(results)
    except Exception as e:
        print(e, "prediation failed")
    # Convert results to JSON
    price_predictions = [{'price': row.prediction, 'airline': AIRLINES[int(row.first_segmentsAirlineName_index)]} for row in results]
    
    price_predictions.sort(key=lambda x: x["airline"])
    stacked_price_predictions = {airline: list(group) for airline, group in groupby(price_predictions, key=lambda x: x["airline"])}
    min_price_predictions = [{'airline': airline, 'price': min(item["price"] for item in group)} for airline, group in stacked_price_predictions.items()]

    return min_price_predictions


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


    try:
        data_df_2 = spark.createDataFrame(pd.DataFrame(data))
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
    print(predictions_2.select('prediction', 'Operating_Airline_index', 'depDaypart_index', 'arrDayPart_index').toPandas())
    
    # Extract prediction results
    try:
        results_2 = predictions_2.select('prediction', 'Operating_Airline_index', 'depDaypart_index', 'arrDayPart_index').collect()
        print(results_2)
    except Exception as e:
        print(e, "prediation failed")
    # Convert results to JSON
    delay_prediction_2 = [{'delay': (row.prediction % 25)  * random.uniform(-1, 1), 
                           'airline': list(ALL_AIRLINES_WITH_CODES.keys())[int(row.Operating_Airline_index)]} 
                           for row in results_2]
    # res = (delay_prediction_2 % 25)  * random.uniform(-1, 1)
    return delay_prediction_2
# 480.995284





@app.route('/getAirportObjects', methods=["GET"])
def getAirportObjects():  
    airportObjects = getAirportObjectsFromJSON()
    return airportObjects


@app.route('/getDelaysAndPrices', methods=["POST"])
def getDelaysAndPrices():
    userData = request.get_json()
    pricesDataList, delaysDataList = buildDataLists(userData).values()
    predictedPrices = predict_prices(pricesDataList)
    predictedDelays = predict_delays(delaysDataList)
    return {"prices": predictedPrices, "delays": predictedDelays}

    #     return {
    #         "delays": delays,
    #         "prices": prices
    #     }


if __name__ == "__main__":
    # To run the Flask app
    app.run(port=8000, debug=True)
