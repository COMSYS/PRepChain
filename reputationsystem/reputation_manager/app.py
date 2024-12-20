import requests
from flask import Flask, jsonify, request
from pymongo import MongoClient
import tomli
import numpy as np
from Pyfhel import Pyfhel, PyCtxt
import os
from os.path import isfile
from ecdsa import SigningKey, VerifyingKey, NIST192p
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
log_handler_pm = RotatingFileHandler(os.path.join(log_dir, 'rm.log'), maxBytes=1024*1024, backupCount=10)
log_handler_pm.setLevel(log_level)
log_handler_pm.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
app.logger.addHandler(log_handler_pm)

client = MongoClient(config["ReputationManager"]["db_host"], config["ReputationManager"]["db_port"])

db = client[config["ReputationManager"]["db"]]
collection = db[config["ReputationManager"]["collection"]]

def create_keydirectory(votee):
    # Get the current directory where app.py is located
    current_directory = os.path.dirname(os.path.abspath(__file__))

    # Create the path for the new directory within the 'keys' directory
    new_directory_path = os.path.join(current_directory, 'keys', votee)

    # Check if the 'keys' directory exists, create it if not
    keys_directory = os.path.join(current_directory, 'keys')
    if not os.path.exists(keys_directory):
        os.makedirs(keys_directory)

    # Check if the new directory already exists, create it if not
    if not os.path.exists(new_directory_path):
        os.makedirs(new_directory_path)
        print(f"Created directory: {new_directory_path}")
        return True, new_directory_path
    else:
        print(f"Directory already exists: {new_directory_path}")
        return False, new_directory_path


def get_keydirectory_path(votee):
    # Get the current directory where app.py is located
    current_directory = os.path.dirname(os.path.abspath(__file__))

    # Create the path for the specified directory within the 'keys' directory
    directory_path = os.path.join(current_directory, 'keys', votee)

    return directory_path

def compute_aggregated_reputation(sub_rating, sub_count):
    weights = config["Rating"]["rating_weights"] # [0.5,0.8,1] -> [l,m,h]

    #Einzelgewichtung
    weighted_sub = []
    weighted_count = []

    #Einzelgewichtung & Vektor-faktoren
    for i in range(config["Rating"]["eq_classes_num"]):
        weight = weights[i]
        start_index = i * config["Rating"]["sub_num"]
        end_index = (i + 1) * config["Rating"]["sub_num"]
        sub_values = sub_rating[start_index:end_index]
        count_values = sub_count[start_index:end_index]
        weighted_sub_tmp = [weight * value for value in sub_values]
        weighted_count_tmp = [weight * value for value in count_values]
        weighted_sub.extend(weighted_sub_tmp)
        weighted_count.extend(weighted_count_tmp)

    r_sum = []
    f_sum = []
    aggr_reputation = []

    for i in range(config["Rating"]["sub_num"]):
        total_rating = 0
        total_count = 0
        index = i
        for j in range(config["Rating"]["eq_classes_num"]):
            total_rating += weighted_sub[index]
            total_count += weighted_count[index]
            index += config["Rating"]["sub_num"]
        r_sum.append(total_rating)
        f_sum.append(total_count)

    for k in range(config["Rating"]["sub_num"]):
        if f_sum[k] != 0: #Division by Zero Check!
            aggr_reputation.append(r_sum[k] / f_sum[k])

    avg_aggr_reputation = sum(aggr_reputation) / config["Rating"]["sub_num"]

    return avg_aggr_reputation, r_sum, f_sum

def enc_compute_aggregated_reputation(user, sub_rating, sub_count):
    key_dir = get_keydirectory_path(str(user))
    HE_server = Pyfhel()  # Empty creation
    HE_server.load_context(key_dir + "/context")
    HE_server.load_public_key(key_dir + "/pub.key")
    #HE_server.load_secret_key(key_dir + "/sec.key")
    HE_server.load_relin_key(key_dir + "/relin.key")
    HE_server.load_rotate_key(key_dir + "/rotate.key")
    enc_subrating = PyCtxt(pyfhel=HE_server,
                                bytestring=sub_rating.encode('cp437'))
    enc_subcount = PyCtxt(pyfhel=HE_server,
                               bytestring=sub_count.encode('cp437'))
    weights = config["Rating"]["rating_weights"] # [0.5,0.8,1] -> [l,m,h]
    padded_weights = np.array([weight for weight in weights for _ in range(config['Rating']['sub_num'])])
    pTxTweights = HE_server.encodeFrac(padded_weights)
    #sub_numOfFields = config["Rating"]["sub_num"] * config["Rating"]["eq_classes_num"]

    #Einzelgewichtung & Vektor-faktoren
    weighted_sub = enc_subrating * pTxTweights
    HE_server.relinearize(weighted_sub)
    weighted_count = enc_subcount * pTxTweights

    #Compute Vectors for each sub_feature to extract weighted values for following summation
    mult_list = []
    for i in range(config["Rating"]["sub_num"]):
        sublist = [0] * (config["Rating"]["eq_classes_num"] * config["Rating"]["sub_num"])  # Initialize a sublist of zeros
        for j in range(config["Rating"]["eq_classes_num"]):
            sublist[i + j * config["Rating"]["sub_num"]] = 1.0  # Set the appropriate indices to 1
        mult_list.append(HE_server.encodeFrac(np.array(sublist)))

    #Extract Values for each Feature and Level in separate Lists
    """Performs cumulative addition over the first n_elements of a PyCtxt.

            Runs log2(n_elements) additions and rotations to obtain the cumulative 
            sum in the first element of the result. For correct results use a power
            of 2 for n_elements. If n_elements is 0, it will use the size (nSlots) 
            of the ciphertext."""
    r_sum = []
    f_sum = []
    for sub_list in mult_list:
        r_sum.append(HE_server.cumul_add(weighted_sub * sub_list))
        f_sum.append(HE_server.cumul_add(weighted_count * sub_list))

    r_recent = {
        'r_sum': [x.to_bytes() for x in r_sum],
        'f_sum': [y.to_bytes() for y in f_sum]
    }

    return r_recent

def aggregate_rating(prev_subrating, recent_subrating, HE_server, prev_enc_subrating, prev_enc_subcount, enc_subrating, enc_subcount):

    new_rating = {
        "sub": {
            "rating": [x + y for x, y in zip(prev_subrating["sub"]["rating"], recent_subrating["sub"]["rating"])],
            "count": [x + y for x, y in zip(prev_subrating["sub"]["count"], recent_subrating["sub"]["count"])]
        },
        "enc_sub": {
            "enc_rating": HE_server.add(prev_enc_subrating,enc_subrating, in_new_ctxt=True).to_bytes().decode('cp437'),
            "enc_count": HE_server.add(prev_enc_subcount,enc_subcount, in_new_ctxt=True).to_bytes().decode('cp437')
        },
        "obj": [x + y for x, y in zip(prev_subrating["obj"], recent_subrating["obj"])]
    }

    return new_rating

def retrieve_keys(user):
    key_dir = get_keydirectory_path(str(user))
    HE_server = Pyfhel()  # Empty creation
    HE_server.load_context(key_dir + "/context")
    HE_server.load_public_key(key_dir + "/pub.key")
    HE_server.load_relin_key(key_dir + "/relin.key")
    HE_server.load_rotate_key(key_dir + "/rotate.key")
    #HE_server.load_secret_key(key_dir + "/sec.key")

    s_context = HE_server.to_bytes_context()
    s_public_key = HE_server.to_bytes_public_key()
    s_relin_key = HE_server.to_bytes_relin_key()
    s_rotate_key = HE_server.to_bytes_rotate_key()
    #s_secret_key = HE_server.to_bytes_secret_key()

    enc_json = {
        'context': s_context.decode('cp437'),
        'pk': s_public_key.decode('cp437'),
        'rlk': s_relin_key.decode('cp437'),
        'rtk': s_rotate_key.decode('cp437'),
        #'sk': s_secret_key.decode('cp437'),
    }

    return enc_json

def get_signing_key():
    # Check if private.pem and public.pem already exist
    if not isfile("private.pem") or not isfile("public.pem"):
        sk = SigningKey.generate()
        vk = sk.verifying_key
        if not isfile("private.pem"):
            with open("private.pem", "wb") as f:
                f.write(sk.to_pem())
        if not isfile("public.pem"):
            with open("public.pem", "wb") as f:
                f.write(vk.to_pem())
    else:
        with open("private.pem") as f:
            sk = SigningKey.from_pem(f.read())
        with open("public.pem") as f:
            vk = VerifyingKey.from_pem(f.read())

    return sk, vk

@app.route('/verifying-key', methods=["POST", "GET"])
def send_vk():
    sk, vk = get_signing_key()
    return jsonify({'vk':vk.to_string().decode('cp437')})

@app.route('/rrecent', methods=["POST", "GET"])
def send_rrecent():
    res = request.get_json()
    sk, vk = get_signing_key()
    pseudonym = res["pseudonym"]
    user = res["user"]

    performance_dict = {}
    performance_dict['eval'] = res['eval']
    performance_dict['iteration'] = res['iteration']
    performance_dict['route'] = '/rrecent'

    # Check if the file exists
    file_path = os.path.join(eval_dir, res['eval'], 'reputation_manager_' + str(res['eval']) + '.csv')
    file_exists = os.path.isfile(file_path)

    t1 = time.perf_counter()

    if pseudonym[0] != "R":
        return jsonify({"error": "Invalid pseudonym",}), 400
    query = collection.find_one({'SystemID': str(user)})

    if query is None:
        return jsonify({"error": "Invalid provided system ID",}), 400

    elif query["total_count"] == 0:  # No reputation for systemID stored, yet
        default_R_recent = 1 # Set initial reputation to 1, new user -> the lowest trust level
        tracemalloc.start()
        start_memory = tracemalloc.get_traced_memory()[0]
        signature = sk.sign(default_R_recent.to_bytes(2, 'little'))
        end_memory = tracemalloc.get_traced_memory()[0]
        tracemalloc.stop()
        memory_usage_diff = end_memory - start_memory
        performance_dict['size_sig'] = memory_usage_diff
        params = {"R_recent": default_R_recent.to_bytes(2, 'little').decode('cp437'), "signature": signature.decode('cp437'), "user": query["SystemID"], "total_count": query["total_count"]}
    elif query["recent_count"] == 0: #Only R_General stored for user
        avg_aggr_reputation, r_sum, f_sum = compute_aggregated_reputation(query["R_general"]["sub"]["rating"], query["R_general"]["sub"]["count"])
        plain_r_recent = {'avg_aggr_reputation': avg_aggr_reputation, 'r_sum': r_sum, 'f_sum':f_sum}
        
        t3 = time.perf_counter()
        tracemalloc.start()
        start_memory = tracemalloc.get_traced_memory()[0]
        r_recent = enc_compute_aggregated_reputation(user, query["R_general"]["enc_sub"]["enc_rating"],
                                                     query["R_general"]["enc_sub"]["enc_count"])
        end_memory = tracemalloc.get_traced_memory()[0]
        tracemalloc.stop()
        t4 = time.perf_counter()
        performance_dict['time_enc_compute_aggregated_reputation'] = round(t4 - t3, 2)
        memory_usage_diff = end_memory - start_memory
        performance_dict['size_r_recent'] = memory_usage_diff
        
        t5 = time.perf_counter()
        tracemalloc.start()
        start_memory = tracemalloc.get_traced_memory()[0]
        r_sum_signs = [sk.sign(x).decode('cp437') for x in r_recent['r_sum']]
        end_memory = tracemalloc.get_traced_memory()[0]
        tracemalloc.stop()
        memory_usage_diff = end_memory - start_memory
        performance_dict['size_r_sum_signs'] = memory_usage_diff
        t6 = time.perf_counter()
        t7 = time.perf_counter()
        tracemalloc.start()
        start_memory = tracemalloc.get_traced_memory()[0]
        f_sum_signs = [sk.sign(y).decode('cp437') for y in r_recent['f_sum']]
        end_memory = tracemalloc.get_traced_memory()[0]
        tracemalloc.stop()
        memory_usage_diff = end_memory - start_memory
        performance_dict['size_f_sum_signs'] = memory_usage_diff
        t8 = time.perf_counter()
        performance_dict['time_r_sum_signs'] = round(t6 - t5, 2)
        performance_dict['time_f_sum_signs'] = round(t8 - t7, 2)

        r_recent['r_sum'] = [x.decode('cp437') for x in r_recent['r_sum']]
        r_recent['f_sum'] = [y.decode('cp437') for y in r_recent['f_sum']]
        enc_keys_json = retrieve_keys(user)
        
        print("Get Secret Key from Inquired")
        enc_sk = requests.post(
            url="http://127.0.0.1:" + str(config["Votee"]["flask_port"]) + "/get-enc-secret-key",
            json={'votee': user, 'eval': res['eval'], 'iteration': res['iteration']}).json()
        params = {"R_recent": r_recent, "r_sum_signs": r_sum_signs, "f_sum_signs": f_sum_signs,
                  "user": query["SystemID"], "enc_keys": enc_keys_json, "total_count": query["total_count"], 'enc_sk':enc_sk, 'plain_r_recent':plain_r_recent}

    elif "R_recent3" in query and query["R_recent3"] is not None:
        avg_aggr_reputation, r_sum, f_sum = compute_aggregated_reputation(query["R_recent3"]["sub"]["rating"], query["R_recent3"]["sub"]["count"])
        plain_r_recent = {'avg_aggr_reputation': avg_aggr_reputation, 'r_sum': r_sum, 'f_sum':f_sum}

        t3 = time.perf_counter()
        tracemalloc.start()
        start_memory = tracemalloc.get_traced_memory()[0]
        r_recent = enc_compute_aggregated_reputation(user, query["R_recent3"]["enc_sub"]["enc_rating"],
                                                     query["R_recent3"]["enc_sub"]["enc_count"])
        end_memory = tracemalloc.get_traced_memory()[0]
        tracemalloc.stop()
        memory_usage_diff = end_memory - start_memory
        performance_dict['size_r_recent'] = memory_usage_diff                                             
        t4 = time.perf_counter()
        performance_dict['time_enc_compute_aggregated_reputation'] = round(t4 - t3, 2)

        t5 = time.perf_counter()
        tracemalloc.start()
        start_memory = tracemalloc.get_traced_memory()[0]
        r_sum_signs = [sk.sign(x).decode('cp437') for x in r_recent['r_sum']]
        end_memory = tracemalloc.get_traced_memory()[0]
        tracemalloc.stop()
        memory_usage_diff = end_memory - start_memory
        performance_dict['size_r_sum_signs'] = memory_usage_diff
        t6 = time.perf_counter()
        t7 = time.perf_counter()
        tracemalloc.start()
        start_memory = tracemalloc.get_traced_memory()[0]
        f_sum_signs = [sk.sign(y).decode('cp437') for y in r_recent['f_sum']]
        end_memory = tracemalloc.get_traced_memory()[0]
        tracemalloc.stop()
        memory_usage_diff = end_memory - start_memory
        performance_dict['size_f_sum_signs'] = memory_usage_diff
        t8 = time.perf_counter()
        performance_dict['time_r_sum_signs'] = round(t6 - t5, 2)
        performance_dict['time_f_sum_signs'] = round(t8 - t7, 2)

        r_recent['r_sum'] = [x.decode('cp437') for x in r_recent['r_sum']]
        r_recent['f_sum'] = [y.decode('cp437') for y in r_recent['f_sum']]
        enc_keys_json = retrieve_keys(user)
        enc_sk = requests.post(
            url="http://127.0.0.1:" + str(config["Votee"]["flask_port"]) + "/get-enc-secret-key",
            json={'votee': user, 'eval': res['eval'], 'iteration': res['iteration']}).json()
        params = {"R_recent": r_recent, "r_sum_signs": r_sum_signs, "f_sum_signs": f_sum_signs,
                  "user": query["SystemID"], "enc_keys": enc_keys_json, "total_count": query["total_count"], 'enc_sk':enc_sk, 'plain_r_recent':plain_r_recent}

    elif "R_recent2" in query and query["R_recent2"] is not None:
        avg_aggr_reputation, r_sum, f_sum = compute_aggregated_reputation(query["R_recent2"]["sub"]["rating"], query["R_recent2"]["sub"]["count"])
        plain_r_recent = {'avg_aggr_reputation': avg_aggr_reputation, 'r_sum': r_sum, 'f_sum':f_sum}
        
        t3 = time.perf_counter()
        tracemalloc.start()
        start_memory = tracemalloc.get_traced_memory()[0]
        r_recent = enc_compute_aggregated_reputation(user, query["R_recent2"]["enc_sub"]["enc_rating"],
                                                     query["R_recent2"]["enc_sub"]["enc_count"])
        end_memory = tracemalloc.get_traced_memory()[0]
        tracemalloc.stop()
        memory_usage_diff = end_memory - start_memory
        performance_dict['size_r_recent'] = memory_usage_diff
        t4 = time.perf_counter()
        performance_dict['time_enc_compute_aggregated_reputation'] = round(t4 - t3, 2)
        
        t5 = time.perf_counter()
        tracemalloc.start()
        start_memory = tracemalloc.get_traced_memory()[0]
        r_sum_signs = [sk.sign(x).decode('cp437') for x in r_recent['r_sum']]
        end_memory = tracemalloc.get_traced_memory()[0]
        tracemalloc.stop()
        memory_usage_diff = end_memory - start_memory
        performance_dict['size_r_sum_signs'] = memory_usage_diff
        t6 = time.perf_counter()
        t7 = time.perf_counter()
        tracemalloc.start()
        start_memory = tracemalloc.get_traced_memory()[0]
        f_sum_signs = [sk.sign(y).decode('cp437') for y in r_recent['f_sum']]
        end_memory = tracemalloc.get_traced_memory()[0]
        tracemalloc.stop()
        memory_usage_diff = end_memory - start_memory
        performance_dict['size_f_sum_signs'] = memory_usage_diff
        t8 = time.perf_counter()
        performance_dict['time_r_sum_signs'] = round(t6 - t5, 2)
        performance_dict['time_f_sum_signs'] = round(t8 - t7, 2)
        r_recent['r_sum'] = [x.decode('cp437') for x in r_recent['r_sum']]
        r_recent['f_sum'] = [y.decode('cp437') for y in r_recent['f_sum']]
        enc_keys_json = retrieve_keys(user)
        enc_sk = requests.post(
            url="http://127.0.0.1:" + str(config["Votee"]["flask_port"]) + "/get-enc-secret-key",
            json={'votee': user, 'eval': res['eval'], 'iteration': res['iteration']}).json()
        params = {"R_recent": r_recent, "r_sum_signs": r_sum_signs, "f_sum_signs": f_sum_signs,
                  "user": query["SystemID"], "enc_keys": enc_keys_json, "total_count": query["total_count"], 'enc_sk':enc_sk, 'plain_r_recent':plain_r_recent}

    elif "R_recent1" in query and query["R_recent1"] is not None:
        avg_aggr_reputation, r_sum, f_sum = compute_aggregated_reputation(query["R_recent1"]["sub"]["rating"], query["R_recent1"]["sub"]["count"])
        plain_r_recent = {'avg_aggr_reputation': avg_aggr_reputation, 'r_sum': r_sum, 'f_sum':f_sum}
        
        t3 = time.perf_counter()
        tracemalloc.start()
        start_memory = tracemalloc.get_traced_memory()[0]
        r_recent = enc_compute_aggregated_reputation(user, query["R_recent1"]["enc_sub"]["enc_rating"], query["R_recent1"]["enc_sub"]["enc_count"])
        end_memory = tracemalloc.get_traced_memory()[0]
        tracemalloc.stop()
        memory_usage_diff = end_memory - start_memory
        performance_dict['size_r_recent'] = memory_usage_diff
        t4 = time.perf_counter()
        performance_dict['time_enc_compute_aggregated_reputation'] = round(t4 - t3, 2)
        
        t5 = time.perf_counter()
        tracemalloc.start()
        start_memory = tracemalloc.get_traced_memory()[0]
        r_sum_signs = [sk.sign(x).decode('cp437') for x in r_recent['r_sum']]
        end_memory = tracemalloc.get_traced_memory()[0]
        tracemalloc.stop()
        memory_usage_diff = end_memory - start_memory
        performance_dict['size_r_sum_signs'] = memory_usage_diff
        t6 = time.perf_counter()
        t7 = time.perf_counter()
        tracemalloc.start()
        start_memory = tracemalloc.get_traced_memory()[0]
        f_sum_signs = [sk.sign(y).decode('cp437') for y in r_recent['f_sum']]
        end_memory = tracemalloc.get_traced_memory()[0]
        tracemalloc.stop()
        memory_usage_diff = end_memory - start_memory
        performance_dict['size_f_sum_signs'] = memory_usage_diff
        t8 = time.perf_counter()
        performance_dict['time_r_sum_signs'] = round(t6 - t5, 2)
        performance_dict['time_f_sum_signs'] = round(t8 - t7, 2)
        r_recent['r_sum'] = [x.decode('cp437') for x in r_recent['r_sum']]
        r_recent['f_sum'] = [y.decode('cp437') for y in r_recent['f_sum']]
        enc_keys_json = retrieve_keys(user)
        enc_sk = requests.post(
            url="http://127.0.0.1:" + str(config["Votee"]["flask_port"]) + "/get-enc-secret-key",
            json={'votee': user, 'eval': res['eval'], 'iteration': res['iteration']}).json()
        params = {"R_recent": r_recent, "r_sum_signs": r_sum_signs , "f_sum_signs": f_sum_signs, "user": query["SystemID"], "enc_keys": enc_keys_json, "total_count": query["total_count"], 'enc_sk':enc_sk, 'plain_r_recent':plain_r_recent}

    t2 = time.perf_counter()
    performance_dict['time'] = round(t2 - t1, 2)

    known_columns = ['eval', 'iteration', 'route', 'size_sig', 'time', 'time_enc_compute_aggregated_reputation', 'size_r_recent', 'size_r_sum_signs', 'size_f_sum_signs', 'time_r_sum_signs', 'time_f_sum_signs', 'size_r_general', 'size_new_rating', 'time_aggregate_rating']
    df_iteration = pd.DataFrame(columns=known_columns)
    data_dict = {col: [performance_dict.get(col, None)] for col in known_columns}
    df_to_append = pd.DataFrame(data_dict)
    df_iteration = pd.concat([df_iteration, df_to_append], ignore_index=True, sort=False)

    df_iteration.to_csv(file_path, mode='a', header=not file_exists, index=False)

    return jsonify(params), 200

@app.route('/rgeneral', methods=["POST", "GET"])
def send_rgeneral():
    res = request.get_json()
    pseudonym = res["pseudonym"]
    inquired = res["inquired"]

    performance_dict = {}
    performance_dict['eval'] = res['eval']
    performance_dict['iteration'] = res['iteration']
    performance_dict['route'] = '/rgeneral'

    # Check if the file exists
    file_path = os.path.join(eval_dir, res['eval'], 'reputation_manager_' + str(res['eval']) + '.csv')
    file_exists = os.path.isfile(file_path)

    t1 = time.perf_counter()

    if pseudonym[0] != "G":
        return jsonify({"error": "Invalid pseudonym",}), 400
    query = collection.find_one({'SystemID': str(inquired)})
    if query is None:
        return jsonify({"error": "Invalid provided system ID",}), 400
    elif query["total_count"] <= config["ReputationManager"]["rating_limit"]: #No G_recent for inquired stored, yet
        params = {"Error": "Inquired does not possess R_general, yet.", "inquired" : query["SystemID"]}
    else:
        avg_aggr_reputation, r_sum, f_sum = compute_aggregated_reputation(query["R_general"]["sub"]["rating"],
                                                            query["R_general"]["sub"]["count"])
        plain_r_general = {'avg_aggr_reputation': avg_aggr_reputation, 'r_sum': r_sum, 'f_sum':f_sum}

        t3 = time.perf_counter()
        tracemalloc.start()
        start_memory = tracemalloc.get_traced_memory()[0]
        r_general = enc_compute_aggregated_reputation(inquired, query["R_general"]["enc_sub"]["enc_rating"],
                                                     query["R_general"]["enc_sub"]["enc_count"])
        end_memory = tracemalloc.get_traced_memory()[0]
        tracemalloc.stop()
        memory_usage_diff = end_memory - start_memory
        performance_dict['size_r_general'] = memory_usage_diff
        t4 = time.perf_counter()
        performance_dict['time_enc_compute_aggregated_reputation'] = round(t4 - t3, 2)

        enc_keys_json = retrieve_keys(inquired)
        r_general['r_sum'] = [x.decode('cp437') for x in r_general['r_sum']]
        r_general['f_sum'] = [y.decode('cp437') for y in r_general['f_sum']]
        print("Get Secret Key from Inquired")
        enc_sk = requests.post(
            url="http://127.0.0.1:" + str(config["Votee"]["flask_port"]) + "/get-enc-secret-key",
            json={'votee': inquired, 'eval': res['eval'], 'iteration': res['iteration']}).json()
        params = {"R_general": r_general, "user": query["SystemID"], "enc_keys": enc_keys_json, "total_count": query["total_count"], 'enc_sk': enc_sk, 'plain_r_general':plain_r_general}
        #params = {"R_general": r_general, "user": query["SystemID"], "enc_keys": enc_keys_json, "total_count": query["total_count"]}

        t2 = time.perf_counter()
        performance_dict['time'] = round(t2 - t1, 2)
        known_columns = ['eval', 'iteration', 'route', 'size_sig', 'time', 'time_enc_compute_aggregated_reputation', 'size_r_recent', 'size_r_sum_signs', 'size_f_sum_signs', 'time_r_sum_signs', 'time_f_sum_signs', 'size_r_general', 'size_new_rating', 'time_aggregate_rating']
        df_iteration = pd.DataFrame(columns=known_columns)
        data_dict = {col: [performance_dict.get(col, None)] for col in known_columns}
        df_to_append = pd.DataFrame(data_dict)
        df_iteration = pd.concat([df_iteration, df_to_append], ignore_index=True, sort=False)

        df_iteration.to_csv(file_path, mode='a', header=not file_exists, index=False)

    return jsonify(params), 200

@app.route('/submit-rating', methods=["POST", "GET"])
def store_rating():
    res = request.get_json()
    pseudonym = res["pseudonym"]
    votee = int(res["votee"])
    subrating = res["subrating"]
    subcount = res["subcount"]
    objrating = res["objrating"]
    upd_rep = ""

    performance_dict = {}
    performance_dict['eval'] = res['eval']
    performance_dict['iteration'] = res['iteration']
    performance_dict['route'] = '/submit-rating'
    
    # Check if the file exists
    file_path = os.path.join(eval_dir, res['eval'], 'reputation_manager_' + str(res['eval']) + '.csv')
    file_exists = os.path.isfile(file_path)

    t1 = time.perf_counter()

    db_entry = collection.find_one({'SystemID': str(votee)})
    if db_entry is None:
        return jsonify({"error": "Invalid system ID",}), 400

    rating_dict = {"sub": {}, "enc_sub": {},"obj": ""}
    rating_dict["sub"]["rating"] = subrating
    rating_dict["sub"]["count"] = subcount
    rating_dict["enc_sub"]["enc_rating"] = res["enc_json"]['s_enc_rating']
    rating_dict["enc_sub"]["enc_count"] = res["enc_json"]['s_enc_count']
    rating_dict["obj"] = objrating

    #if votee does not have R_recent and R_general
    if db_entry["total_count"] == 0:
        if "enc_json" not in res.keys():
            return jsonify({"error": "no encryption material provided"}), 400

        key_directory_created, key_dir_path = create_keydirectory(str(res["votee"]))
        if not key_directory_created:
            return jsonify({"error": "keys for votee already stored"}), 400

        # Read all bytestrings
        HE_server = Pyfhel()
        HE_server.from_bytes_context(res["enc_json"]['context'].encode('cp437'))
        HE_server.from_bytes_public_key(res["enc_json"]['pk'].encode('cp437'))
        #HE_server.from_bytes_secret_key(res["enc_json"]['sk'].encode('cp437'))
        HE_server.from_bytes_relin_key(res["enc_json"]['rlk'].encode('cp437'))
        HE_server.from_bytes_rotate_key(res["enc_json"]['rtk'].encode('cp437'))
        """enc_subrating = PyCtxt(pyfhel=HE_server, bytestring=res["enc_json"]('s_enc_rating').encode('cp437'))
        enc_subcount = PyCtxt(pyfhel=HE_server, bytestring=res["enc_json"]('s_enc_count').encode('cp437'))
        print(f"[Server] received HE_server={HE_server} and cx={enc_subrating}")"""

        # Now we save all objects into files
        HE_server.save_context(key_dir_path + "/context")
        HE_server.save_public_key(key_dir_path + "/pub.key")
        #HE_server.save_secret_key(key_dir_path + "/sec.key") #Just for testing
        HE_server.save_relin_key(key_dir_path + "/relin.key")
        HE_server.save_rotate_key(key_dir_path + "/rotate.key")

        collection.update_one({'SystemID': str(votee)}, {'$set': {'R_recent1': rating_dict, 'recent_count': 1, 'total_count': 1}, '$push': {'voter_pseudonyms': pseudonym}})
        upd_rep = "R_recent1"

    #if votee has R_general but no R_recent
    elif db_entry['recent_count'] == 0:
        key_dir = get_keydirectory_path(str(res["votee"]))
        HE_server = Pyfhel()  # Empty creation
        HE_server.load_context(key_dir + "/context")
        HE_server.load_public_key(key_dir + "/pub.key")
        HE_server.load_relin_key(key_dir + "/relin.key")
        HE_server.load_rotate_key(key_dir + "/rotate.key")
        prev_enc_subrating = PyCtxt(pyfhel=HE_server,
                                    bytestring=db_entry["R_general"]["enc_sub"]['enc_rating'].encode('cp437'))
        prev_enc_subcount = PyCtxt(pyfhel=HE_server,
                                   bytestring=db_entry["R_general"]["enc_sub"]['enc_count'].encode('cp437'))
        enc_subrating = PyCtxt(pyfhel=HE_server, bytestring=res["enc_json"]['s_enc_rating'].encode('cp437'))
        enc_subcount = PyCtxt(pyfhel=HE_server, bytestring=res["enc_json"]['s_enc_count'].encode('cp437'))
        prev_rating = db_entry["R_general"]

        t3 = time.perf_counter()
        tracemalloc.start()
        start_memory = tracemalloc.get_traced_memory()[0]
        new_rating = aggregate_rating(prev_rating, rating_dict, HE_server, prev_enc_subrating, prev_enc_subcount,
                                      enc_subrating, enc_subcount)
        end_memory = tracemalloc.get_traced_memory()[0]
        tracemalloc.stop()
        memory_usage_diff = end_memory - start_memory
        performance_dict['size_new_rating'] = memory_usage_diff
        t4 = time.perf_counter()
        performance_dict['time_aggregate_rating'] = round(t4 - t3, 2)

        collection.update_one({'SystemID': str(votee)}, {'$set': {'R_recent1': new_rating}, '$inc': {'total_count': 1, 'recent_count': 1}, '$push': {'voter_pseudonyms': pseudonym}})
        upd_rep = "R_recent1"

    elif db_entry['recent_count'] == 3:
        key_dir = get_keydirectory_path(str(res["votee"]))
        HE_server = Pyfhel()  # Empty creation
        HE_server.load_context(key_dir + "/context")
        HE_server.load_public_key(key_dir + "/pub.key")
        HE_server.load_relin_key(key_dir + "/relin.key")
        HE_server.load_rotate_key(key_dir + "/rotate.key")
        prev_enc_subrating = PyCtxt(pyfhel=HE_server,
                                    bytestring=db_entry["R_recent3"]["enc_sub"]['enc_rating'].encode('cp437'))
        prev_enc_subcount = PyCtxt(pyfhel=HE_server,
                                   bytestring=db_entry["R_recent3"]["enc_sub"]['enc_count'].encode('cp437'))
        enc_subrating = PyCtxt(pyfhel=HE_server, bytestring=res["enc_json"]['s_enc_rating'].encode('cp437'))
        enc_subcount = PyCtxt(pyfhel=HE_server, bytestring=res["enc_json"]['s_enc_count'].encode('cp437'))
        prev_rating = db_entry["R_recent3"]
        
        t3 = time.perf_counter()
        tracemalloc.start()
        start_memory = tracemalloc.get_traced_memory()[0]
        new_rating = aggregate_rating(prev_rating, rating_dict, HE_server, prev_enc_subrating, prev_enc_subcount,
                                      enc_subrating, enc_subcount)
        end_memory = tracemalloc.get_traced_memory()[0]
        tracemalloc.stop()
        memory_usage_diff = end_memory - start_memory
        performance_dict['size_new_rating'] = memory_usage_diff
        t4 = time.perf_counter()
        performance_dict['time_aggregate_rating'] = round(t4 - t3, 2)

        collection.update_one({'SystemID': str(votee)}, {'$set': {'R_general': new_rating, 'R_recent1': None, 'R_recent2': None, 'R_recent3': None, 'recent_count': 0}, '$inc': {'total_count': 1}, '$push': {'voter_pseudonyms': pseudonym}})
        upd_rep = "R_general"

        # Check if with this R_recent the limit is reached, then overwrite R_general and reset recent count and R_recent
        """if db_entry['recent_count'] == config["ReputationManager"]["rating_limit"]:
            collection.update_one({'SystemID': votee},
                                  {'$set': {'R_general': rating_dict, 'recent_count': 0, 'R_recent': {}},
                                   '$inc': {'total_count': 1}, '$push': {'voter_pseudonyms': pseudonym}})
        else:
            collection.update_one({'SystemID': votee},
                                  {'$set': {'R_recent': rating_dict}, '$inc': {'total_count': 1, 'recent_count': 1},
                                   '$push': {'voter_pseudonyms': pseudonym}})"""
    elif db_entry['recent_count'] == 2:
        key_dir = get_keydirectory_path(str(res["votee"]))
        HE_server = Pyfhel()  # Empty creation
        HE_server.load_context(key_dir + "/context")
        HE_server.load_public_key(key_dir + "/pub.key")
        HE_server.load_relin_key(key_dir + "/relin.key")
        HE_server.load_rotate_key(key_dir + "/rotate.key")
        prev_enc_subrating = PyCtxt(pyfhel=HE_server,
                                    bytestring=db_entry["R_recent2"]["enc_sub"]['enc_rating'].encode('cp437'))
        prev_enc_subcount = PyCtxt(pyfhel=HE_server,
                                   bytestring=db_entry["R_recent2"]["enc_sub"]['enc_count'].encode('cp437'))
        enc_subrating = PyCtxt(pyfhel=HE_server, bytestring=res["enc_json"]['s_enc_rating'].encode('cp437'))
        enc_subcount = PyCtxt(pyfhel=HE_server, bytestring=res["enc_json"]['s_enc_count'].encode('cp437'))
        prev_rating = db_entry["R_recent2"]
        
        t3 = time.perf_counter()
        tracemalloc.start()
        start_memory = tracemalloc.get_traced_memory()[0]
        new_rating = aggregate_rating(prev_rating, rating_dict, HE_server, prev_enc_subrating, prev_enc_subcount,
                                      enc_subrating, enc_subcount)
        end_memory = tracemalloc.get_traced_memory()[0]
        tracemalloc.stop()
        memory_usage_diff = end_memory - start_memory
        performance_dict['size_new_rating'] = memory_usage_diff
        t4 = time.perf_counter()
        performance_dict['time_aggregate_rating'] = round(t4 - t3, 2)

        collection.update_one({'SystemID': str(votee)},
                              {'$set': {'R_recent3': new_rating}, '$inc': {'total_count': 1, 'recent_count': 1},
                               '$push': {'voter_pseudonyms': pseudonym}})       
        upd_rep = "R_recent3"

    elif db_entry['recent_count'] == 1:
        key_dir = get_keydirectory_path(str(res["votee"]))
        HE_server = Pyfhel()  # Empty creation
        HE_server.load_context(key_dir + "/context")
        HE_server.load_public_key(key_dir + "/pub.key")
        HE_server.load_relin_key(key_dir + "/relin.key")
        HE_server.load_rotate_key(key_dir + "/rotate.key")
        prev_enc_subrating = PyCtxt(pyfhel=HE_server, bytestring=db_entry["R_recent1"]["enc_sub"]['enc_rating'].encode('cp437'))
        prev_enc_subcount = PyCtxt(pyfhel=HE_server, bytestring=db_entry["R_recent1"]["enc_sub"]['enc_count'].encode('cp437'))
        enc_subrating = PyCtxt(pyfhel=HE_server, bytestring=res["enc_json"]['s_enc_rating'].encode('cp437'))
        enc_subcount = PyCtxt(pyfhel=HE_server, bytestring=res["enc_json"]['s_enc_count'].encode('cp437'))
        prev_rating = db_entry["R_recent1"]
        
        t3 = time.perf_counter()
        tracemalloc.start()
        start_memory = tracemalloc.get_traced_memory()[0]
        new_rating = aggregate_rating(prev_rating, rating_dict, HE_server, prev_enc_subrating, prev_enc_subcount, enc_subrating, enc_subcount)
        end_memory = tracemalloc.get_traced_memory()[0]
        tracemalloc.stop()
        memory_usage_diff = end_memory - start_memory
        performance_dict['size_new_rating'] = memory_usage_diff
        t4 = time.perf_counter()
        performance_dict['time_aggregate_rating'] = round(t4 - t3, 2)

        collection.update_one({'SystemID': str(votee)}, {'$set': {'R_recent2': new_rating}, '$inc': {'total_count': 1, 'recent_count': 1}, '$push': {'voter_pseudonyms': pseudonym}})
        upd_rep = "R_recent2"

    t2 = time.perf_counter()
    performance_dict['time'] = round(t2 - t1, 2)
    known_columns = ['eval', 'iteration', 'route', 'size_sig', 'time', 'time_enc_compute_aggregated_reputation', 'size_r_recent', 'size_r_sum_signs', 'size_f_sum_signs', 'time_r_sum_signs', 'time_f_sum_signs', 'size_r_general', 'size_new_rating', 'time_aggregate_rating']
    df_iteration = pd.DataFrame(columns=known_columns)
    data_dict = {col: [performance_dict.get(col, None)] for col in known_columns}
    df_to_append = pd.DataFrame(data_dict)
    df_iteration = pd.concat([df_iteration, df_to_append], ignore_index=True, sort=False)

    df_iteration.to_csv(file_path, mode='a', header=not file_exists, index=False)

    return jsonify({"Success": "New Rating stored and aggregated", "upd_rep": upd_rep}), 200

if __name__ == '__main__':
   app.run(debug=config["DEBUG"]["flask_debug"], port=config["ReputationManager"]["flask_port"])