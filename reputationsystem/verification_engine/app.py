import requests
from flask import Flask, jsonify, request
from pymongo import MongoClient
from pymongo import ReturnDocument
import random
import tomli
from ecdsa import SigningKey, VerifyingKey, BadSignatureError, NIST192p
from Pyfhel import Pyfhel, PyCtxt
from os.path import isfile
import numpy as np
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
log_handler_pm = RotatingFileHandler(os.path.join(log_dir, 've.log'), maxBytes=1024*1024, backupCount=10)
log_handler_pm.setLevel(log_level)
log_handler_pm.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
app.logger.addHandler(log_handler_pm)


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

def check_rating_indices(subrating, count, eq_class):
    # Calculate the start and end indices for the subset
    start_idx = eq_class * config["Rating"]["sub_num"]
    end_idx = (eq_class + 1) * config["Rating"]["sub_num"]

    # Check that elements before start_idx are 0
    if any(subrating[:start_idx]) and any(count[:start_idx]):
        return False

    # Check that elements at and after end_idx are 0
    if any(subrating[end_idx:]) and any(count[:start_idx]):
        return False

    # Check that elements within the subset are between 1 and 10
    for idx in range(start_idx, end_idx):
        if (subrating[idx]!= 0 and (subrating[idx] < 1 or subrating[idx] > 10)) and (count[idx] != 0 and count[idx] != 1):
            return False

    return True


def compute_rep_level(HE_client, r_recent):
    r_sum = [PyCtxt(pyfhel=HE_client, bytestring=x.encode('cp437')) for x in r_recent['r_sum']]
    f_sum = [PyCtxt(pyfhel=HE_client, bytestring=x.encode('cp437')) for x in r_recent['f_sum']]

    r_sum = [np.round(HE_client.decryptFrac(x)[:1], 2) for x in r_sum]
    f_sum = [np.round(HE_client.decryptFrac(y)[:1], 2) for y in f_sum]

    aggr_reputation = []
    for k in range(config["Rating"]["sub_num"]):
        if f_sum[k] != 0: #Division by 0 ChecK!
            aggr_reputation.append(r_sum[k] / f_sum[k])

    avg_aggr_reputation = sum(aggr_reputation) / config["Rating"]["sub_num"]

    return avg_aggr_reputation

@app.route('/verifying-key', methods=["POST", "GET"])
def send_vk():
    sk, vk = get_signing_key()
    return jsonify({'vk':vk.to_string().decode('cp437')})

@app.route('/verify-rating-no-enc', methods=["POST", "GET"])
def verify_rating_no_enc():
    res = request.get_json()
    pseudonym = res["pseudonym"]
    eq_val = int.from_bytes(res['eq_val'].encode('cp437'), 'little')
    sign = res['sign'].encode('cp437')
    enc_json = res['enc_json']
    if pseudonym[0] != "R":
        return jsonify({"error": "Invalid pseudonym",}), 400

    performance_dict = {}
    performance_dict['eval'] = res['eval']
    performance_dict['iteration'] = res['iteration']
    performance_dict['route'] = '/verify-rating-no-enc'

    # Check if the file exists
    file_path = os.path.join(eval_dir, res['eval'], 'verification_engine_' + str(res['eval']) + '.csv')
    file_exists = os.path.isfile(file_path)

    t1 = time.perf_counter()

    response = requests.post(url="http://127.0.0.1:"+str(config["ReputationManager"]["flask_port"])+"/verifying-key").json()
    vk = VerifyingKey.from_string(response['vk'].encode('cp437'),curve=NIST192p)

    #Check if reputation level sent by RM is verifiable through verification key
    t3 = time.perf_counter()
    try:
        vk.verify(sign, res['eq_val'].encode('cp437'))
    except BadSignatureError:
        return jsonify({"Error": "Provided signature did not verify reputation level"}), 400
    t4 = time.perf_counter()
    performance_dict['time_verify_eq'] = round(t4 - t3, 2)

    eq_class = ""
    eq_class1 = config["Rating"]["eq_classes"][0]
    eq_class2 = config["Rating"]["eq_classes"][1]
    eq_class3 = config["Rating"]["eq_classes"][2]
    if eq_class1[0] < eq_val <= eq_class1[1]:
        eq_class = 0
    elif eq_class2[0] < eq_val <= eq_class2[1]:
        eq_class = 1
    elif eq_class3[0] < eq_val <= eq_class3[1]:
        eq_class = 2
    else:
        eq_class = 3

    # Read all bytestrings
    HE_server = Pyfhel()
    HE_server.from_bytes_context(enc_json['context'].encode('cp437'))
    HE_server.from_bytes_public_key(enc_json['pk'].encode('cp437'))
    HE_server.from_bytes_secret_key(enc_json['sk'].encode('cp437'))
    enc_subrating = PyCtxt(pyfhel=HE_server, bytestring=enc_json['s_enc_rating'].encode('cp437'))
    enc_count = PyCtxt(pyfhel=HE_server, bytestring=enc_json['s_enc_count'].encode('cp437'))
    subrating = np.round(HE_server.decryptFrac(enc_subrating)[:len(config["Rating"]["rating_fields"])], 1)
    count = np.round(HE_server.decryptFrac(enc_count)[:len(config["Rating"]["rating_fields"])], 1)

    t5 = time.perf_counter()
    if check_rating_indices(subrating.tolist(), count.tolist(), eq_class):
        sk, vk = get_signing_key()
        tracemalloc.start()
        start_memory = tracemalloc.get_traced_memory()[0]
        signature_rating = sk.sign(enc_json['s_enc_rating'].encode('cp437'))
        end_memory = tracemalloc.get_traced_memory()[0]
        tracemalloc.stop()
        memory_usage_diff = end_memory - start_memory
        performance_dict['size_sign_rating'] = memory_usage_diff

        tracemalloc.start()
        start_memory = tracemalloc.get_traced_memory()[0]
        signature_count = sk.sign(enc_json['s_enc_count'].encode('cp437'))
        end_memory = tracemalloc.get_traced_memory()[0]
        tracemalloc.stop()
        memory_usage_diff = end_memory - start_memory
        performance_dict['size_sign_count'] = memory_usage_diff
        params = {'signature_rating': signature_rating.decode('cp437'), 'signature_count': signature_count.decode('cp437')}
    else:
        return jsonify({"Error": "Invalid Rating submitted"}), 400
    t6 = time.perf_counter()
    performance_dict['time_sign_rating'] = round(t6 - t5, 2)

    t2 = time.perf_counter()
    performance_dict['time'] = round(t2 - t1, 2)

    known_columns = ['eval', 'iteration', 'route', 'time', 'time_verify_eq', 'size_sign_rating', 'size_sign_count', 'time_sign_rating', 'time_verify_rep_lvl', 'time_compute_rep_lvl']
    df_iteration = pd.DataFrame(columns=known_columns)
    data_dict = {col: [performance_dict.get(col, None)] for col in known_columns}
    df_to_append = pd.DataFrame(data_dict)
    df_iteration = pd.concat([df_iteration, df_to_append], ignore_index=True, sort=False)

    df_iteration.to_csv(file_path, mode='a', header=not file_exists, index=False)

    return jsonify(params), 200

@app.route('/verify-rating', methods=["POST", "GET"])
def verify_rating():
    res = request.get_json()
    pseudonym = res["pseudonym"]
    r_recent = res['r_recent']
    r_sum_signs = res['r_sum_signs']
    f_sum_signs = res['f_sum_signs']
    votee_enc_json = res['votee_enc_json']
    voter_enc_json = res['voter_enc_json']
    if pseudonym[0] != "R":
        return jsonify({"error": "Invalid pseudonym",}), 400

    performance_dict = {}
    performance_dict['eval'] = res['eval']
    performance_dict['iteration'] = res['iteration']
    performance_dict['route'] = '/verify-rating'

    # Check if the file exists
    file_path = os.path.join(eval_dir, res['eval'], 'verification_engine_' + str(res['eval']) + '.csv')
    file_exists = os.path.isfile(file_path)

    t1 = time.perf_counter()

    #Get Verifying Key from RM
    response = requests.post(url="http://127.0.0.1:"+str(config["ReputationManager"]["flask_port"])+"/verifying-key").json()
    vk = VerifyingKey.from_string(response['vk'].encode('cp437'),curve=NIST192p)

    #Check if r_sum and f_sum sent by RM is verifiable through verification key
    t3 = time.perf_counter()
    try:
        for i in range(min(len(r_recent['r_sum']), len(r_sum_signs))):
            vk.verify(r_sum_signs[i].encode('cp437'),r_recent['r_sum'][i].encode('cp437'))
            vk.verify(f_sum_signs[i].encode('cp437'),r_recent['f_sum'][i].encode('cp437'))
    except BadSignatureError:
        return jsonify({"Error": "Provided signature did not verify reputation level"}), 400
    t4 = time.perf_counter()
    performance_dict['time_verify_rep_lvl'] = round(t4 - t3, 2)

    HE_voter = Pyfhel()
    HE_voter.from_bytes_context(voter_enc_json["context"].encode('cp437'))
    HE_voter.from_bytes_public_key(voter_enc_json["pk"].encode('cp437'))
    HE_voter.from_bytes_relin_key(voter_enc_json["rlk"].encode('cp437'))
    HE_voter.from_bytes_rotate_key(voter_enc_json["rtk"].encode('cp437'))
    HE_voter.from_bytes_secret_key(voter_enc_json['sk'].encode('cp437'))

    t5 = time.perf_counter()
    avg_aggr_rep_level = compute_rep_level(HE_voter, r_recent)
    t6 = time.perf_counter()
    performance_dict['time_compute_rep_lvl'] = round(t6 - t5, 2)

    eq_class1 = config["Rating"]["eq_classes"][0]
    eq_class2 = config["Rating"]["eq_classes"][1]
    eq_class3 = config["Rating"]["eq_classes"][2]
    if eq_class1[0] <= avg_aggr_rep_level < eq_class1[1]:
        eq_class = 0
    elif eq_class2[0] <= avg_aggr_rep_level < eq_class2[1]:
        eq_class = 1
    elif eq_class3[0] <= avg_aggr_rep_level <= eq_class3[1]:
        eq_class = 2
    else:
        eq_class = 3

    # Read all bytestrings
    HE_votee = Pyfhel()
    HE_votee.from_bytes_context(votee_enc_json["context"].encode('cp437'))
    HE_votee.from_bytes_public_key(votee_enc_json["pk"].encode('cp437'))
    HE_votee.from_bytes_secret_key(votee_enc_json['sk'].encode('cp437'))
    enc_subrating = PyCtxt(pyfhel=HE_votee, bytestring=voter_enc_json['s_enc_rating'].encode('cp437'))
    enc_count = PyCtxt(pyfhel=HE_votee, bytestring=voter_enc_json['s_enc_count'].encode('cp437'))
    subrating = np.round(HE_votee.decryptFrac(enc_subrating)[:len(config["Rating"]["rating_fields"])], 1)
    count = np.round(HE_votee.decryptFrac(enc_count)[:len(config["Rating"]["rating_fields"])], 1)

    t7 = time.perf_counter()
    if check_rating_indices(subrating.tolist(), count.tolist(), eq_class):
        sk, vk = get_signing_key()
        tracemalloc.start()
        start_memory = tracemalloc.get_traced_memory()[0]
        signature_rating = sk.sign(voter_enc_json['s_enc_rating'].encode('cp437'))
        end_memory = tracemalloc.get_traced_memory()[0]
        tracemalloc.stop()
        memory_usage_diff = end_memory - start_memory
        performance_dict['size_sign_rating'] = memory_usage_diff

        tracemalloc.start()
        start_memory = tracemalloc.get_traced_memory()[0]
        signature_count = sk.sign(voter_enc_json['s_enc_count'].encode('cp437'))
        end_memory = tracemalloc.get_traced_memory()[0]
        tracemalloc.stop()
        memory_usage_diff = end_memory - start_memory
        performance_dict['size_sign_rating'] = memory_usage_diff
        params = {'signature_rating': signature_rating.decode('cp437'), 'signature_count': signature_count.decode('cp437')}

    else:
        return jsonify({"Error": "Invalid Rating submitted"}), 400
    t8 = time.perf_counter()
    performance_dict['time_sign_rating'] = round(t6 - t5, 2)

    t2 = time.perf_counter()
    performance_dict['time'] = round(t2 - t1, 2)

    known_columns = ['eval', 'iteration', 'route', 'time', 'time_verify_eq', 'size_sign_rating', 'size_sign_count', 'time_sign_rating', 'time_verify_rep_lvl', 'time_compute_rep_lvl']
    df_iteration = pd.DataFrame(columns=known_columns)
    data_dict = {col: [performance_dict.get(col, None)] for col in known_columns}
    df_to_append = pd.DataFrame(data_dict)
    df_iteration = pd.concat([df_iteration, df_to_append], ignore_index=True, sort=False)

    df_iteration.to_csv(file_path, mode='a', header=not file_exists, index=False)

    return jsonify(params), 200

if __name__ == '__main__':
   app.run(debug=config["DEBUG"]["flask_debug"], port=config["VerificationEngine"]["flask_port"])