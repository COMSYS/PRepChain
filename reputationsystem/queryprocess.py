import argparse
import sys
import numpy as np
import pandas as pd
import tomli
from Pyfhel import Pyfhel, PyCtxt
import requests
from pymongo import MongoClient
import os
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.fernet import Fernet
import base64
import logging
from logging.handlers import RotatingFileHandler
import random
import cProfile
import time
from memory_profiler import profile, memory_usage
import tracemalloc
import pickle

from interface.interface import request_recentpseudonym, request_generalpseudonym, request_rrecent, request_rgeneral, register_vote, send_ratinginfo, send_for_verification_no_enc, send_for_verification, send_secret_key, get_secret_key, request_pk, request_sk

with open("config.toml", mode="rb") as fp:
    config = tomli.load(fp)

# Define the log file directory path (parent_directory/logging)
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logging')

# Ensure the log directory exists, create it if it doesn't
os.makedirs(log_dir, exist_ok=True)

# Get the log level from the configuration
log_level = config["Logging"]["log_level"]

# Logging configuration
log_handler = RotatingFileHandler(os.path.join(log_dir, 'query.log'), maxBytes=1024*1024, backupCount=10)
log_handler.setLevel(log_level)
log_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))

# Create a logger and add the log handler to it
logger = logging.getLogger('my_logger')
logger.setLevel(log_level)
logger.addHandler(log_handler)

def main(iteration, eval_run, inquired):
    t0 = time.perf_counter(), time.process_time()
    performance_dict = {}
    performance_dict['eval'] = eval_run
    performance_dict['iteration'] = [iteration]

    parser = argparse.ArgumentParser(
        prog="Rating Process",
        description="A script to simulate the rating process in our reputation system")
    parser.add_argument(
        "-bid",
        "--BusinessID",
        type=int,
        default=1,
        help="insert system ID here (default: %(default)s)",)
    parser.add_argument(
        "-i",
        "--inquiredSystemID",
        type=int,
        default=inquired,
        help="insert system ID of inquired here (default: %(default)s)", )

    args = parser.parse_args()

    performance_dict['inquired'] = args.inquiredSystemID

    print("Request Pseudonym to query R_general")
    t1 = time.perf_counter(), time.process_time()
    pseudonym_request = request_generalpseudonym(args.BusinessID, eval_run, iteration).json()
    t2 = time.perf_counter(), time.process_time()
    performance_dict['pseudonym_request_perf'] = round(t2[0] - t1[0], 2)
    #performance_dict['encryption_pseudonym_proc'] = round(t2[1] - t1[1], 2)

    print("Request Pseudonym for encrypted secret key")
    t1 = time.perf_counter(), time.process_time()
    encryption_pseudonym = request_generalpseudonym(args.BusinessID, eval_run, iteration).json()
    t2 = time.perf_counter(), time.process_time()
    performance_dict['encryption_pseudonym_perf'] = round(t2[0] - t1[0], 2)
    #performance_dict['encryption_pseudonym_proc'] = round(t2[1] - t1[1], 2)

    t1 = time.perf_counter(), time.process_time()
    reputation_request = request_rgeneral(pseudonym_request["generalpseudonym"], args.inquiredSystemID, eval_run, iteration).json()
    t2 = time.perf_counter(), time.process_time()
    performance_dict['reputation_request_perf'] = round(t2[0] - t1[0], 2)
    #performance_dict['encryption_pseudonym_proc'] = round(t2[1] - t1[1], 2)

    if "R_general" not in reputation_request.keys():
        return print({"Error": "Votee has no R_general"})

    he_enc_keys = reputation_request['enc_keys']
    inq_enc_keys = reputation_request["enc_sk"]
    #print(inq_enc_keys.keys())
    t1 = time.perf_counter(), time.process_time()
    enc_sk_request = request_sk(encryption_pseudonym["generalpseudonym"], args.inquiredSystemID, eval_run, iteration).json()
    t2 = time.perf_counter(), time.process_time()
    performance_dict['enc_sk_request_perf'] = round(t2[0] - t1[0], 2)
    #performance_dict['encryption_pseudonym_proc'] = round(t2[1] - t1[1], 2)
    #print(enc_sk_request.keys())

    # Deserialize the private key
    t1 = time.perf_counter(), time.process_time()
    private_key = serialization.load_pem_private_key(
        base64.b64decode(enc_sk_request['private_key']),  # Ensure it's encoded as bytes
        password=None,  # If your private key is password-protected, provide the password here
        backend=default_backend()
    )
    t2 = time.perf_counter(), time.process_time()
    performance_dict['private_key_perf'] = round(t2[0] - t1[0], 2)
    #performance_dict['encryption_pseudonym_proc'] = round(t2[1] - t1[1], 2)

    t1 = time.perf_counter(), time.process_time()
    fernet_key = private_key.decrypt(
        base64.b64decode(inq_enc_keys["enc_fernet_key"]),
        padding.OAEP(
            mgf=padding.MGF1(algorithm=padding.hashes.SHA256()),
            algorithm=padding.hashes.SHA256(),
            label=None
        )
    )
    t2 = time.perf_counter(), time.process_time()
    performance_dict['fernet_key_perf'] = round(t2[0] - t1[0], 2)
    #performance_dict['encryption_pseudonym_proc'] = round(t2[1] - t1[1], 2)

    t1 = time.perf_counter(), time.process_time()
    fernet_cipher = Fernet(fernet_key)
    t2 = time.perf_counter(), time.process_time()
    performance_dict['fernet_cipher_perf'] = round(t2[0] - t1[0], 2)
    #performance_dict['encryption_pseudonym_proc'] = round(t2[1] - t1[1], 2)
    t1 = time.perf_counter(), time.process_time()
    decrypted_sk = fernet_cipher.decrypt(base64.b64decode(inq_enc_keys["fernet_enc_sk"]))
    t2 = time.perf_counter(), time.process_time()
    performance_dict['decrypted_sk_perf'] = round(t2[0] - t1[0], 2)
    #performance_dict['encryption_pseudonym_proc'] = round(t2[1] - t1[1], 2)

    #inq_enc_keys = reputation_request["enc_keys"]

    #print("Get Secret Key from Votee")
    #sk_get = get_secret_key(args.inquiredSystemID).json()

    t1 = time.perf_counter(), time.process_time()
    HE_client = Pyfhel()
    HE_client.from_bytes_context(he_enc_keys['context'].encode('cp437'))
    HE_client.from_bytes_public_key(he_enc_keys['pk'].encode('cp437'))
    HE_client.from_bytes_relin_key(he_enc_keys['rlk'].encode('cp437'))
    HE_client.from_bytes_rotate_key(he_enc_keys['rtk'].encode('cp437'))
    HE_client.from_bytes_secret_key(decrypted_sk)
    t2 = time.perf_counter(), time.process_time()
    performance_dict['HE_client_perf'] = round(t2[0] - t1[0], 2)
    #performance_dict['encryption_pseudonym_proc'] = round(t2[1] - t1[1], 2)

    t1 = time.perf_counter(), time.process_time()
    avg_aggr_rep_lvl, aggr_rep, r_sum, f_sum = compute_rep_level(HE_client, reputation_request["R_general"], reputation_request["plain_r_general"])
    t2 = time.perf_counter(), time.process_time()
    performance_dict['avg_aggr_rep_lvl_perf'] = round(t2[0] - t1[0], 2)
    #performance_dict['encryption_pseudonym_proc'] = round(t2[1] - t1[1], 2)

    t100 = time.perf_counter(), time.process_time()
    performance_dict['query_process_perf'] = round(t100[0] - t0[0], 2)
    performance_dict['query_process_proc'] = round(t100[1] - t0[1], 2)

    #print(f"Business {args.inquiredSystemID} has average aggregated reputation level of {avg_aggr_rep_lvl} with r_sum/f_sum : {r_sum}/{f_sum} and R={aggr_rep}")

    return avg_aggr_rep_lvl, aggr_rep, r_sum, f_sum, performance_dict

def compute_rep_level(HE_client, r_recent, plain_r_recent):
    r_sum = [PyCtxt(pyfhel=HE_client, bytestring=x.encode('cp437')) for x in r_recent['r_sum']]
    f_sum = [PyCtxt(pyfhel=HE_client, bytestring=x.encode('cp437')) for x in r_recent['f_sum']]

    r_sum = [np.round(HE_client.decryptFrac(x)[:1], 2) for x in r_sum]
    f_sum = [np.round(HE_client.decryptFrac(y)[:1], 2) for y in f_sum]

    aggr_reputation = []
    for k in range(config["Rating"]["sub_num"]):
        if f_sum[k] != 0: #Division by 0 ChecK!
            aggr_reputation.append(r_sum[k] / f_sum[k])

    avg_aggr_reputation = sum(aggr_reputation) / config["Rating"]["sub_num"]

    plain_r_sum = plain_r_recent["r_sum"]
    plain_f_sum = plain_r_recent["f_sum"]
    plain_avg_aggr_reputation = plain_r_recent["avg_aggr_reputation"]

    # Check if values at corresponding indices are the same
    for i in range(min(len(r_sum), len(plain_r_sum))):
        if np.round(r_sum[i],1) != round(plain_r_sum[i],1):
            print(f"Values at index {i} are different: {r_sum[i]} and {plain_r_sum[i]}")
        
        """if r_sum[i] == plain_r_sum[i]:
            print(f"Values at index {i} are the same: {r_sum[i]}")
        else:
            print(f"Values at index {i} are different: {r_sum[i]} and {plain_r_sum[i]}")"""

    for i in range(min(len(f_sum), len(plain_f_sum))):
        if np.round(f_sum[i],1) != round(plain_f_sum[i],1):
            print(f"Values at index {i} are different: {f_sum[i]} and {plain_f_sum[i]}")
        
        """if f_sum[i] == plain_f_sum[i]:
            print(f"Values at index {i} are the same: {f_sum[i]}")
        else:
            print(f"Values at index {i} are different: {f_sum[i]} and {plain_f_sum[i]}")"""

    if avg_aggr_reputation != plain_avg_aggr_reputation:
        print(f"WARNING: Average reputation DIFFERENT")

    return avg_aggr_reputation, aggr_reputation, r_sum, f_sum

if __name__ == "__main__":
    #main()
    # Define the evaluation name
    evaluation_name = 'eval5'

    # Define the path to the "eval_data" folder
    eval_data_folder = os.path.join(os.getcwd(), 'eval_data')

    # Create a folder for the evaluation if it doesn't exist within "eval_data"
    evaluation_folder = os.path.join(eval_data_folder, evaluation_name)
    os.makedirs(evaluation_folder, exist_ok=True)

    # Create an empty DataFrame outside the loop
    result_df = pd.DataFrame(columns=['Iteration'])  # Initialize with 'Iteration' column
    # Check if the file exists within the evaluation folder
    file_path = os.path.join(evaluation_folder, 'query_process.csv')
    file_exists = os.path.isfile(file_path)
    
    if file_exists:
        df = pd.read_csv(file_path)  
        last_iteration_str = df['iteration'].iloc[-1]  # get the last iteration value
        last_iteration_num = int(last_iteration_str.strip('[]'))  # remove the brackets and convert to int
        iteration = last_iteration_num + 1  # increment by 1 for the next iteration
    else:
        iteration = 0

    for run in range(0,20):
        inquired = 20102 #Eval run x, inquired dann 20"x"02
        avg_aggr_rep_lvl, aggr_rep, r_sum, f_sum, performance_dict = main(iteration, evaluation_name, inquired)
        
        iteration += 1

        performance_dict['run'] = run

        known_columns = ['eval',
                        'iteration',
                        'run',
                        'inquired',
                        'pseudonym_request_perf',
                        'encryption_pseudonym_perf',
                        'reputation_request_perf',
                        'enc_sk_request_perf',
                        'private_key_perf',
                        'fernet_key_perf',
                        'fernet_cipher_perf',
                        'decrypted_sk_perf',
                        'HE_client_perf',
                        'avg_aggr_rep_lvl_perf',
                        'query_process_perf',
                        'query_process_proc']
        df_iteration = pd.DataFrame(columns=known_columns)
        data_dict = {col: [performance_dict.get(col, None)] for col in known_columns}
        df_to_append = pd.DataFrame(data_dict)
        df_iteration = pd.concat([df_iteration, df_to_append], ignore_index=True, sort=False)

        # Save the df_iteration DataFrame to the CSV file (append mode if the file exists)
        df_iteration.to_csv(file_path, mode='a', header=not file_exists, index=False)

        # After writing the CSV for the first time, set file_exists to True
        file_exists = True      