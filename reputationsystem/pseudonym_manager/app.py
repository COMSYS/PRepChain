import requests
from flask import Flask, jsonify, request
from pymongo import MongoClient
from pymongo import ReturnDocument
import random
import tomli
import os
import logging
from logging.handlers import RotatingFileHandler
import pandas as pd
import time
import tracemalloc

with open("../config.toml", mode="rb") as fp:
    config = tomli.load(fp)

app = Flask(__name__)

# Define the log file directory path (parent_directory/logging)
log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logging')
eval_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'eval_data')

# Ensure the log directory exists, create it if it doesn't
os.makedirs(log_dir, exist_ok=True)
os.makedirs(eval_dir, exist_ok=True)

# Get the log level from the configuration
log_level = config["Logging"]["log_level"]
app.logger.setLevel(log_level)

# Logging configuration for Pseudonym Manager
log_handler_pm = RotatingFileHandler(os.path.join(log_dir, 'pm.log'), maxBytes=1024*1024, backupCount=10)
log_handler_pm.setLevel(log_level)
log_handler_pm.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
app.logger.addHandler(log_handler_pm)

client = MongoClient(config["PseudonymManager"]["db_host"], config["PseudonymManager"]["db_port"])

db = client[config["PseudonymManager"]["db"]]
collection = db[config["PseudonymManager"]["collection"]]

@app.route('/recent-pseudonym', methods=["POST", "GET"])
def create_recentpseudonym():
    res = request.get_json()

    performance_dict = {}
    performance_dict['eval'] = res['eval']
    performance_dict['iteration'] = [res['iteration']]
    performance_dict['route'] = '/recent-pseudonym'

    # Check if the file exists
    file_path = os.path.join(eval_dir, res['eval'], 'pseudonym_manager_' + str(res['eval']) + '.csv')
    file_exists = os.path.isfile(file_path)

    t1 = time.perf_counter()

    pseudonym = "R" + str(random.randint(1000, 2000))
    query = collection.find_one_and_update({'BusinessID': str(res["systemid"])}, {'$push': {'Recent': [pseudonym]}})
    if query is None:
        app.logger.error(f'Did not find entry in collection for BusinessID: {str(res["systemid"])}')
        return jsonify({"error": "Invalid Business ID"}), 400
    params = {"recentpseudonym": pseudonym, "systemid": res["systemid"]}
    app.logger.debug(f"params for endpoint /recent-pseudonym: {params}")

    t2 = time.perf_counter()
    performance_dict['time'] = round(t2 - t1, 2)
    df_iteration = pd.DataFrame(performance_dict)

    df_iteration.to_csv(file_path, mode='a', header=not file_exists, index=False)

    return jsonify(params), 200

@app.route('/general-pseudonym', methods=["POST", "GET"])
def create_generalpseudonym():
    res = request.get_json()

    performance_dict = {}
    performance_dict['eval'] = res['eval']
    performance_dict['iteration'] = [res['iteration']]
    performance_dict['route'] = '/general-pseudonym'

    # Check if the file exists
    file_path = os.path.join(eval_dir, res['eval'], 'pseudonym_manager_' + str(res['eval']) + '.csv')
    file_exists = os.path.isfile(file_path)

    t1 = time.perf_counter()
    
    pseudonym = "G"+str(random.randint(1000,2000))
    query = collection.find_one_and_update({'BusinessID': str(res["systemid"])}, {'$push': {'General': [pseudonym]}})
    if query is None:
        app.logger.error(f'Did not find entry in collection for BusinessID: {str(res["systemid"])}')
        return jsonify({"error": "Invalid Business ID"}), 400
    params = {"generalpseudonym": pseudonym, "systemid": res["systemid"]}
    app.logger.debug(f"params for endpoint /general-pseudonym: {params}")

    t2 = time.perf_counter()
    performance_dict['time'] = round(t2 - t1, 2)
    df_iteration = pd.DataFrame(performance_dict)

    df_iteration.to_csv(file_path, mode='a', header=not file_exists, index=False)

    return jsonify(params), 200


if __name__ == '__main__':
   app.run(debug=config["DEBUG"]["flask_debug"], port=config["PseudonymManager"]["flask_port"])
