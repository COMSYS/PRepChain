import requests
from flask import Flask, jsonify, request
from pymongo import MongoClient
from pymongo import ReturnDocument
import random
import datetime
import tomli
import requests
from bson.objectid import ObjectId
import ast
import sys
import json
from ecdsa import SigningKey, VerifyingKey, BadSignatureError, NIST192p
import os
import logging
from logging.handlers import RotatingFileHandler
import time
import sys
import pandas as pd
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
log_handler_pm = RotatingFileHandler(os.path.join(log_dir, 're.log'), maxBytes=1024*1024, backupCount=10)
log_handler_pm.setLevel(log_level)
log_handler_pm.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
app.logger.addHandler(log_handler_pm)

client = MongoClient(config["ReputationEngine"]["db_host"], config["ReputationEngine"]["db_port"])

db = client[config["ReputationEngine"]["db"]]
collection = db[config["ReputationEngine"]["collection"]]

@app.route('/register-vote', methods=["POST", "GET"])
def create_vote_entry():
    res = request.get_json()

    performance_dict = {}
    performance_dict['eval'] = res['eval']
    performance_dict['iteration'] = res['iteration']
    performance_dict['route'] = '/register-vote'

    # Check if the file exists
    file_path = os.path.join(eval_dir, res['eval'], 'reputation_engine_' + str(res['eval']) + '.csv')
    file_exists = os.path.isfile(file_path)

    t1 = time.perf_counter()

    tracemalloc.start()
    start_memory = tracemalloc.get_traced_memory()[0]
    entry = {
        "voter": res['voterid'],
        "votee": res['voteeid'],
        "rating_num": res['rating_num'],
        "rating_fields": res['rating_fields'],
        "rating_types": res['rating_types'],
        "rating_limits": res['rating_limits']
    }
    end_memory = tracemalloc.get_traced_memory()[0]
    tracemalloc.stop()
    memory_usage_diff = end_memory - start_memory
    performance_dict['size_register_vote'] = memory_usage_diff

    t3 = time.perf_counter()
    query = collection.insert_one(entry)
    t4 = time.perf_counter()
    performance_dict['time_insert_registered_vote'] = round(t4 - t3, 2)
    if query is None:
        return jsonify({"Error": "Not able to register vote, please try again."}), 400

    t2 = time.perf_counter()
    performance_dict['time'] = round(t2 - t1, 2)

    known_columns = ['eval', 'iteration', 'route', 'time', 'proc_time', 'size_register_vote','time_insert_registered_vote', 'time_verify_rating', 'size_obj_rating', 'time_gen_obj_rating']
    df_iteration = pd.DataFrame(columns=known_columns)
    data_dict = {col: [performance_dict.get(col, None)] for col in known_columns}
    df_to_append = pd.DataFrame(data_dict)
    df_iteration = pd.concat([df_iteration, df_to_append], ignore_index=True, sort=False)

    df_iteration.to_csv(file_path, mode='a', header=not file_exists, index=False)

    return jsonify({"Success": "Vote registered", "voteid": str(query.inserted_id)}), 200

def compute_objrating(rating_specs, obj_fields, spec, res, eval, iteration):
    index = obj_fields.index(spec)

    votee_data = {'reputation_engine_id': config["ReputationEngine"]["id"], 'voter': res["voterid"],
                  'measure': spec, 'eval': eval, 'iteration': iteration}
    votee_response = requests.post(url="http://127.0.0.1:"+str(config["Votee"]["flask_port"])+"/obj-data", json=votee_data).json()

    if votee_response["data_exists"] != 1: #Check if votee provides specific data
        return 0
    else:
        if spec.startswith("trusted"): #Check if trusted data
            response = requests.post(
                url="http://127.0.0.1:" + str(config["Votee"]["flask_port"]) + "/verifying-key").json()
            vk = VerifyingKey.from_string(response['vk'].encode('cp437'), curve=NIST192p)

            # Check if measurements sent by Votee are correctly signed
            try:
                vk.verify(votee_response['signature'].encode('cp437'),
                          json.dumps(votee_response['measurements']).encode('cp437'))
            except BadSignatureError:
                return jsonify({"Error": "Provided signature did not verify measurements"}), 400

        if rating_specs["rating_types"][index] == "diff":
            limit = rating_specs["rating_limits"][index]
            stop_val = (limit / 100) * votee_response["num_of_datapoints"] #set stop value to the percentage of all datapoints provided
            counter = 0
            for entry in votee_response["measurements"]:
                if votee_response["measurements"][entry][0] != votee_response["measurements"][entry][1]:
                    counter += 1
                    if counter >= stop_val:
                        return 0
        elif isinstance(rating_specs["rating_types"][index], list):
            interval = rating_specs["rating_types"][index]
            limit = rating_specs["rating_limits"][index]
            stop_val = (limit / 100) * votee_response[
                "num_of_datapoints"]  # set stop value to the percentage of all datapoints provided
            counter = 0
            for entry in votee_response["measurements"]:
                #print(votee_response["measurements"][entry])
                if not interval[0] <= float(votee_response["measurements"][entry]) <= interval[1] + 1:
                    #print(votee_response["measurements"][entry], interval[0], interval[1] + 1)
                    counter += 1
                    if counter >= stop_val:
                        return 0
        """elif isinstance(rating_specs["rating_types"][index], list): #IN CASE WE USE CSV TO SUBMIT SENSOR DATA INSTEAD OF JSON
            interval = rating_specs["rating_types"][index]
            limit = rating_specs["rating_limits"][index]
            counter = 0
            for entry in votee_response["measurements"]:
                if int(votee_response["measurements"][entry][spec]) not in range(interval[0], interval[1] + 1):
                    counter += 1
                    if counter >= limit:
                        return 0"""
    return 1

def compute_overall_rating(rating_fields, subrating, objrating, voter_rrecent, votee_rrecent):

    #subrating = ast.literal_eval(subrating) # subrating looks originally like this "[1,1]"
    rating = subrating + objrating

    rating_dict = {}
    for key in rating_fields:
        for value in rating:
            rating_dict[key] = value
            rating.remove(value)
            break

    return rating_dict

@app.route('/send-ratinginfo', methods=["POST", "GET"])
def calc_rating():
    res = request.get_json()
    sign_rating = res['signature_rating'].encode('cp437')
    sign_count = res['signature_count'].encode('cp437')

    #Check if subjective or objective rating already submitted for specific vote id
    db_entry = collection.find_one({"_id": ObjectId(res["voteid"])})
    if 'objrating' in db_entry:
        return jsonify(
            {"Error": ("Database with id {} already stores an objective rating.").format(res["voteid"])}), 400
    elif 'subrating' in db_entry:
        return jsonify(
            {"Error": ("Database with id {} already stores an subjective rating.").format(res["voteid"])}), 400

    performance_dict = {}
    performance_dict['eval'] = res['eval']
    performance_dict['iteration'] = res['iteration']
    performance_dict['route'] = '/send-ratinginfo'

    # Check if the file exists
    file_path = os.path.join(eval_dir, res['eval'], 'reputation_engine_' + str(res['eval']) + '.csv')
    file_exists = os.path.isfile(file_path)

    t1 = time.perf_counter(), time.process_time()

    info = {
        "subrating": res['subrating'],
        "subcount": res['subcount'],
        "voter_rrecent": res['rrecent'],
        "pseudonym": res['pseudonym'],
        "voteid": res["voteid"]
    }

    response = requests.post(
        url="http://127.0.0.1:" + str(config["VerificationEngine"]["flask_port"]) + "/verifying-key").json()
    vk = VerifyingKey.from_string(response['vk'].encode('cp437'), curve=NIST192p)

    t3 = time.perf_counter()
    # Check if reputation level sent by RM is verifiable through verification key
    try:
        vk.verify(sign_rating, res['enc_json']['s_enc_rating'].encode('cp437'))
        vk.verify(sign_count, res['enc_json']['s_enc_count'].encode('cp437'))
    except BadSignatureError:
        return jsonify({"Error": "Provided signature did not verify reputation level"}), 400
    t4 = time.perf_counter()
    performance_dict['time_verify_rating'] = round(t4 - t3, 2)

    # Insert Rating info from voter for later use
    inserted_query = collection.update_one({"voter": res["voterid"], "votee": res["voteeid"], "_id": ObjectId(res["voteid"])}, {"$set": info})
    if inserted_query is None:
        return jsonify({"error": "Not able to store rating info, please try again."}), 400

    # Retrieve rating specifications from previous registration of vote
    rating_specs = collection.find_one({"voter": res['voterid'], "votee": res['voteeid'], "_id": ObjectId(res["voteid"])})
    if rating_specs is None:
        return jsonify({"error: encountered issue when querying for existing rating specifications"}), 400

    t5 = time.perf_counter()
    # Calculate objective rating entries, skip subjective fields in rating specifications
    obj_index_start = rating_specs["rating_num"][0]
    obj_fields = rating_specs["rating_fields"][obj_index_start:]

    # for every rating field after subjective rating fields, compute PADDED objective rating
    obj_rating = []
    tracemalloc.start()
    start_memory = tracemalloc.get_traced_memory()[0]
    obj_rating = [compute_objrating(rating_specs, obj_fields, spec, res, res['eval'], res['iteration']) if spec in obj_fields else 0 for spec in config['Rating']['rating_fields'][config['Rating']['sub_num']:]]
    end_memory = tracemalloc.get_traced_memory()[0]
    tracemalloc.stop()
    memory_usage_diff = end_memory - start_memory
    performance_dict['size_obj_rating'] = memory_usage_diff
    t6 = time.perf_counter()
    performance_dict['time_gen_obj_rating'] = round(t6 - t5, 2)

    obj_spec = collection.update_one({"voter": res["voterid"], "votee": res["voteeid"], "_id": ObjectId(res["voteid"])}, {"$set": {"objrating": obj_rating}})
    if obj_spec.modified_count != 1:
        return jsonify({"Error": "Objective rating not updated."}), 400

    updated_db_entry = collection.find_one({"_id": ObjectId(res["voteid"])})

    submitted_rating = {'pseudonym': updated_db_entry["pseudonym"], 'votee': updated_db_entry["votee"], 'subrating': updated_db_entry["subrating"], 'subcount': updated_db_entry["subcount"], "objrating": updated_db_entry["objrating"], "enc_json": res["enc_json"], 'eval': res['eval'], 'iteration': res['iteration']}
    response = requests.post(url="http://127.0.0.1:"+str(config["ReputationManager"]["flask_port"])+"/submit-rating", json=submitted_rating).json()

    t2 = time.perf_counter(), time.process_time()
    performance_dict['time'] = round(t2[0] - t1[0], 2)
    performance_dict['proc_time'] = round(t2[1] - t1[1], 2)

    known_columns = ['eval', 'iteration', 'route', 'time', 'proc_time', 'size_register_vote','time_insert_registered_vote', 'time_verify_rating', 'size_obj_rating', 'time_gen_obj_rating']
    df_iteration = pd.DataFrame(columns=known_columns)
    data_dict = {col: [performance_dict.get(col, None)] for col in known_columns}
    df_to_append = pd.DataFrame(data_dict)
    df_iteration = pd.concat([df_iteration, df_to_append], ignore_index=True, sort=False)

    df_iteration.to_csv(file_path, mode='a', header=not file_exists, index=False)

    return jsonify({"Success": "Rating submitted", "Updated_Rep": response["upd_rep"]}), 200

if __name__ == '__main__':
   app.run(debug=config["DEBUG"]["flask_debug"], port=config["ReputationEngine"]["flask_port"])