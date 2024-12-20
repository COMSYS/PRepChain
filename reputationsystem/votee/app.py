import requests
from flask import Flask, jsonify, request
from pymongo import MongoClient
from pymongo import ReturnDocument
import random
import datetime
import tomli
import requests
import csv
import os
from Pyfhel import Pyfhel, PyCtxt
from os.path import isfile
from ecdsa import SigningKey, VerifyingKey, NIST192p
from cryptography.fernet import Fernet
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.asymmetric import padding
import json
import time
import sys
import pandas as pd
import tracemalloc

with open("../config.toml", mode="rb") as fp:
    config = tomli.load(fp)

app = Flask(__name__)
app.json.sort_keys = False #IMPORTANT FOR SIGNING TRUSTED DATA JSONIFY CHANGES ORDER OF DICT SO SIGNATURE AT RECEIVER DOES NOT APPLY ANYMORE

# Define the log file directory path (parent_directory/logging)
log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logging')
eval_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'eval_data')

# Ensure the log directory exists, create it if it doesn't
os.makedirs(log_dir, exist_ok=True)
os.makedirs(eval_dir, exist_ok=True)

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

@app.route('/obj-data', methods=["POST", "GET"])
def send_objdata():
    res = request.get_json()
    sk, vk = get_signing_key()
    csv_flag = False
    json_flag = False
    if str(res["reputation_engine_id"]) not in config["Votee"]["allowed_rep_engines"]:
        return jsonify({"error": "query stems from unauthorized party, please try again."}), 400

    performance_dict = {}
    performance_dict['eval'] = res['eval']
    performance_dict['iteration'] = res['iteration']
    performance_dict['route'] = '/obj-data'

    # Check if the file exists
    file_path = os.path.join(eval_dir, res['eval'], 'votee_' + str(res['eval']) + '.csv')
    file_exists = os.path.isfile(file_path)

    t1 = time.perf_counter()

    if not os.path.exists(res['measure'] + "_" + config["Votee"]["csv_file"]) and not os.path.exists(res['measure'] + "_" + config["Votee"]["json_file"]):
        return jsonify({"data_exists": 0}), 201
    elif os.path.exists(res['measure'] + "_" + config["Votee"]["csv_file"]):
        csv_flag = True
    else:
        json_flag = True

    if csv_flag == True:
        t3 = time.perf_counter()
        tracemalloc.start()
        start_memory = tracemalloc.get_traced_memory()[0]
        with open(res['measure'] + "_" + config["Votee"]["csv_file"], encoding='utf-8-sig') as csv_file_handler:
            csv_reader = csv.DictReader(csv_file_handler, delimiter=";")
            data = {"data_exists": 1, "measurements": {}, 'num_of_datapoints':0}
            for rows in csv_reader:
                key = rows['id']
                if res['measure'].startswith('goodsreceipt') or res['measure'].startswith('trustedgoodsreceipt'):
                    data["measurements"][key] = [rows['quantity delivered'], rows['quantity stored']]
                else:
                    data["measurements"][key] = rows
                data['num_of_datapoints'] += 1
        end_memory = tracemalloc.get_traced_memory()[0]
        tracemalloc.stop()
        memory_usage_diff = end_memory - start_memory
        performance_dict['size_prep_csv_data'] = memory_usage_diff
        t4 = time.perf_counter()
        performance_dict['time_prep_csv_data'] = round(t4 - t3, 2)

        t5 = time.perf_counter()
        tracemalloc.start()
        start_memory = tracemalloc.get_traced_memory()[0]
        if res['measure'].startswith("trusted"):
            #data['measurements'] = json.dumps(data['measurements'])
            signature = sk.sign(json.dumps(data['measurements']).encode('cp437'))
            data['signature'] = signature.decode('cp437')
        end_memory = tracemalloc.get_traced_memory()[0]
        tracemalloc.stop()
        memory_usage_diff = end_memory - start_memory
        performance_dict['size_sign_csv_data'] = memory_usage_diff
        t6 = time.perf_counter()
        performance_dict['time_sign_csv_data'] = round(t6 - t5, 2)

    elif json_flag == True:
        t7 = time.perf_counter()
        tracemalloc.start()
        start_memory = tracemalloc.get_traced_memory()[0]
        with open(res['measure'] + "_" + config["Votee"]["json_file"], 'r') as json_file:
            json_data = json.load(json_file)
            data = {"data_exists": 1, "measurements": {}, 'num_of_datapoints':0}
            if res['measure'].startswith('temperatursaegeblatt') or res['measure'].startswith('trustedtemperatursaegeblatt'):
                for entry in json_data:
                    key = entry['id']
                    data['measurements'][key] = round((5.625 / 184.5) * entry['pinValue'] - 2.5, 2) #Calculation Pin to Value
                    #print(data['measurements'][key])
                    data['num_of_datapoints'] += 1

            elif res['measure'].startswith('schwingung') or res['measure'].startswith('trustedschwingung'):
                for entry in json_data:
                    key = entry['id']
                    data['measurements'][key] = round((1.5625 / 184.5) * entry['pinValue'] - 6.25, 2) #Calculation Pin to Value
                    #print(data['measurements'][key])
                    data['num_of_datapoints'] += 1

            elif res['measure'].startswith('vortrieb') or res['measure'].startswith('trustedvortrieb'):
                for entry in json_data:
                    key = entry['id']
                    data['measurements'][key] = round((18.75 * (entry['pinValue'] - 1148)) / 0.5, 2) #Calculation Pin to Value
                    #print(data['measurements'][key])
                    data['num_of_datapoints'] += 1
        end_memory = tracemalloc.get_traced_memory()[0]
        tracemalloc.stop()
        memory_usage_diff = end_memory - start_memory
        performance_dict['size_prep_json_data'] = memory_usage_diff
        t8 = time.perf_counter()
        performance_dict['time_prep_json_data'] = round(t8 - t7, 2)


        t9 = time.perf_counter()
        tracemalloc.start()
        start_memory = tracemalloc.get_traced_memory()[0]
        if res['measure'].startswith("trusted"):
            signature = sk.sign(json.dumps(data['measurements']).encode('cp437'))
            data['signature'] = signature.decode('cp437')
        end_memory = tracemalloc.get_traced_memory()[0]
        tracemalloc.stop()
        memory_usage_diff = end_memory - start_memory
        performance_dict['size_sign_json_data'] = memory_usage_diff
        t10 = time.perf_counter()
        performance_dict['time_sign_json_data'] = round(t10 - t9, 2)

    t2 = time.perf_counter()
    performance_dict['time'] = round(t2 - t1, 2)

    known_columns = ['eval', 'iteration', 'route', 'time', 'size_prep_csv_data', 'time_prep_csv_data', 'size_prep_json_data', 'time_prep_json_data', 'size_sign_json_data', 'time_sign_json_data']
    df_iteration = pd.DataFrame(columns=known_columns)
    data_dict = {col: [performance_dict.get(col, None)] for col in known_columns}
    df_to_append = pd.DataFrame(data_dict)
    df_iteration = pd.concat([df_iteration, df_to_append], ignore_index=True, sort=False)

    df_iteration.to_csv(file_path, mode='a', header=not file_exists, index=False)

    return jsonify(data), 200

@app.route('/store-secret-key', methods=["POST", "GET"])
def store_secret_key():
    res = request.get_json()
    sk = res['secret_key'].encode('cp437')
    context = res['context'].encode('cp437')
    pk = res['pk'].encode('cp437')
    HE_server = Pyfhel()
    HE_server.from_bytes_context(context)
    HE_server.from_bytes_secret_key(sk)
    HE_server.from_bytes_public_key(pk)

    key_directory_created, key_dir_path = create_keydirectory(str(res["votee"]))
    if not key_directory_created:
        return jsonify({"error": "keys for votee already stored"}), 205

    else:
        HE_server.save_secret_key(key_dir_path + "/sec.key")
        HE_server.save_context(key_dir_path + "/context")
        HE_server.save_public_key(key_dir_path + "/pub.key")

    return jsonify({'Success': 'Key for Votee stored'}), 200

@app.route('/get-secret-key', methods=["POST", "GET"])
def send_secret_key():
    res = request.get_json()
    key_dir = get_keydirectory_path(str(res['votee']))
    HE_server = Pyfhel()
    HE_server.load_context(key_dir + "/context")
    HE_server.load_secret_key(key_dir + "/sec.key")
    HE_server.load_public_key(key_dir + "/pub.key")
    s_secret_key = HE_server.to_bytes_secret_key()
    params = {'sk': s_secret_key.decode('cp437')}
    return jsonify(params), 200

@app.route('/get-enc-secret-key', methods=["POST", "GET"])
def send_enc_secret_key():
    res = request.get_json()
    key_dir = get_keydirectory_path(str(res['votee']))
    HE_server = Pyfhel()
    HE_server.load_context(key_dir + "/context")
    HE_server.load_secret_key(key_dir + "/sec.key")
    s_secret_key = HE_server.to_bytes_secret_key()
    
    response = requests.post(
        url="http://127.0.0.1:" + str(config["KeyManager"]["flask_port"]) + "/get-public-key", json={'votee': str(res['votee']), 'sk': s_secret_key.decode('cp437'), 'eval': res['eval'], 'iteration': res['iteration']}).json()

    return jsonify(response), 200

if __name__ == '__main__':
   app.run(debug=config["DEBUG"]["flask_debug"], port=config["Votee"]["flask_port"])