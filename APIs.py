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

CANCELLATION_RATES = {
    'UA': 1.9987360577743913, 
    'NK': 2.1692518411749098, 
    'EV': 4.0611419151805785, 
    'B6': 2.6304631043533058, 
    'DL': 1.418366532449831, 
    '9K': 0.12040939193257075, 
    'OO': 2.297287417756781, 
    'F9': 2.4068633294299957, 
    'YV': 3.4394620509973093, 
    'C5': 4.231911031809512, 
    'EM': 5.592076809964536, 
    'HA': 1.012285138778951, 
    'G4': 4.5788720882713525, 
    'ZW': 3.706450781426715, 
    'CP': 1.2814143304190728, 
    'YX': 3.1884297314645744, 
    'AS': 1.888753656515411, 
    'QX': 2.3393674666191235, 
    'G7': 3.341941364119702, 
    'VX': 2.4504810413129596, 
    'WN': 3.1311177477317353, 
    'AX': 4.093075066526394, 
    '9E': 2.0228926573594705, 
    'AA': 3.0330073829407134, 
    'MQ': 3.36723907462681, 
    'OH': 3.3215979607613715, 
    'KS': 15.486884656845131, 
    'PT': 3.949123876625911, 
}

DIVERSION_RATES = {
    'UA': 0.25945641990063445, 
    'NK': 0.18943604232849764, 
    'EV': 0.30254277304485266, 
    'B6': 0.3186029207678656, 
    'DL': 0.18337335963242776, 
    '9K': 0.2408187838651415, 
    'OO': 0.31009439871025035, 
    'F9': 0.13936317166036757, 
    'YV': 0.25092897108444023, 
    'C5': 0.3130191349289362, 
    'EM': 0.45411296600640083, 
    'HA': 0.08945177005103255, 
    'G4': 0.2876992235390274, 
    'ZW': 0.2151487893277784, 
    'CP': 0.1496919056682905, 
    'YX': 0.22933635791428553, 
    'AS': 0.24617686555388693, 
    'QX': 0.23431878816435425, 
    'G7': 0.23871009743712157, 
    'VX': 0.4753820033955857, 
    'WN': 0.18888125123416727, 
    'AX': 0.36945355529426327, 
    '9E': 0.17701437753450128, 
    'AA': 0.24836341463959385, 
    'MQ': 0.24944583129035083, 
    'OH': 0.2588746578633961, 
    'KS': 2.1200143729787997, 
    'PT': 0.2966753483958358, 
}

loaded_pipeline = PipelineModel.load(PRICES_PIPELINE_PATH)
loaded_pipeline_2 = PipelineModel.load(DELAYS_PIPELINE_PATH)
loaded_model_2 = LinearRegressionModel.load(DELAYS_MODEL_PATH)
loaded_model = LinearRegressionModel.load(PRICES_MODEL_PATH)

def getAirportObjectsFromJSON():
    with open('airportCodeToNameMapping.json', 'r') as file:
        data = json.load(file)
    objects = [{'id': key, 'code': key, 'name': value} for key, value in data.items()]
    sorted_objects = sorted(objects, key=lambda x: x['name'])
    return sorted_objects

def buildDataLists(userData):
    data_list = {}
    data_2_list = []
    origin, destination, departureDayParts, arrivalDayParts, date, searchDate = userData.values()

    flightWeek = datetime.fromisoformat(date.split('.')[0]).isocalendar()[1]
    userSearchWeek = datetime.fromisoformat(searchDate.split('.')[0]).isocalendar()[1]
 
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

    with open('timeDuration.json', 'r') as file:
        timeDurations = json.load(file)
    orig_dest = origin['code'] + '_' + destination['code']
    travelDurationMinutes = timeDurations.get(orig_dest, 0.0)

    for airline in AIRLINES:
        for departureDayPart in departureDayParts:  
            for arrivalDayPart in arrivalDayParts:  
                for searchWeek in range(userSearchWeek, flightWeek + 1):
                    data = {
                        "startingAirport": origin['code'],
                        "destinationAirport": destination['code'],
                        "first_segmentsAirlineName": airline,
                        "searchWeek": searchWeek,
                        "flightWeek": flightWeek,
                        "segmentsDepartureDaypart": departureDayPart,
                        "segmentArrivalDaypart": arrivalDayPart,
                        "totalTravelDistance": totalTravelDistance,
                        "travelDurationMinutes": travelDurationMinutes,
                    }
                    if searchWeek in data_list:
                        data_list[searchWeek].append(data)
                    else:
                        data_list[searchWeek] = [data]                    
                    print(searchWeek, data)

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
    except Exception as e:
        print(e, "prediation failed")
    # Convert results to JSON
    price_predictions = [{'price': row.prediction, 'airline': AIRLINES[int(row.first_segmentsAirlineName_index)]} for row in results]
    
    price_predictions.sort(key=lambda x: x["airline"])
    stacked_price_predictions = {airline: list(group) for airline, group in groupby(price_predictions, key=lambda x: x["airline"])}
    min_price_predictions = [{'airline': airline, 'price': min(item["price"] for item in group)} for airline, group in stacked_price_predictions.items()]

    return min_price_predictions


def random_func(prices):
    random.seed(prices)
    return random.uniform(-1,1)

def get_min_price(prediction_list):
    min_price = float('inf')  # Initialize min_price to positive infinity
    min_price_prediction = None
    
    for prediction in prediction_list:
        price = prediction['price']
        if price is not None and price < min_price:
            min_price = price
            min_price_prediction = prediction
            
    return min_price_prediction

def predict_delays(data):
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
    print(predictions_2.select('prediction', 'Operating_Airline_index', 'depDaypart_index').toPandas())
    
    # Extract prediction results
    try:
        results_2 = predictions_2.select('prediction', 'Operating_Airline_index', 'depDaypart_index').collect()
        print(results_2)
    except Exception as e:
        print(e, "prediation failed")
    # Convert results to JSON
    delay_prediction_2 = [{'delay': (row.prediction), 
                           'airline': list(ALL_AIRLINES_WITH_CODES.keys())[int(row.Operating_Airline_index)]} 
                           for row in results_2]
    
    delay_prediction_2.sort(key=lambda x: x["airline"])
    stacked_delay_predictions = {airline: list(group) for airline, group in groupby(delay_prediction_2, key=lambda x: x["airline"])}
    min_delay_predictions = [{'airline': airline, 'delay': min(item["delay"] for item in group)} for airline, group in stacked_delay_predictions.items()]

    return min_delay_predictions


@app.route('/getAirportObjects', methods=["GET"])
def getAirportObjects():  
    airportObjects = getAirportObjectsFromJSON()
    return airportObjects


@app.route('/getDelaysAndPrices', methods=["POST"])
def getDelaysAndPrices():
    userData = request.get_json()
    pricesDataList, delaysDataList = buildDataLists(userData).values()

    predictedPrices = []

    # Dictionary to store the minimum price prediction for each airline
    min_price_predictions = {}

    # Iterate over each search week
    for searchWeek, data in pricesDataList.items():
        price_predictions = predict_prices(data)
        
        # Update the minimum price prediction for each airline
        for prediction in price_predictions:
            airline = prediction['airline']
            price = prediction['price']
            
            # If the airline is not in min_price_predictions or the new price is lower, update the minimum price prediction
            if airline not in min_price_predictions or price < min_price_predictions[airline]['price']:
                min_price_predictions[airline] = {'airline': airline, 'search_week': searchWeek, 'price': price, 'cancellation_rate': CANCELLATION_RATES.get(ALL_AIRLINES_WITH_CODES[airline], None), 'diversion_rate': DIVERSION_RATES.get(ALL_AIRLINES_WITH_CODES[airline], None)}

    # Create a new list containing only the minimum price predictions
    predictedPrices = list(min_price_predictions.values())


    predictedDelays = predict_delays(delaysDataList)
    return {"prices": predictedPrices, "delays": predictedDelays}

if __name__ == "__main__":
    # To run the Flask app
    app.run(port=8000, debug=True)
