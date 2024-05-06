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
CORS(app, resources={r"/*" : {"origins":"*"}})

@app.route('/getCityNames', methods=["GET"])
def getCityNames():
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
