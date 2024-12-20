import requests
from flask import Flask, jsonify, request
from pymongo import MongoClient
from pymongo import ReturnDocument
import random
import tomli
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.fernet import Fernet
import base64
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
log_handler_pm = RotatingFileHandler(os.path.join(log_dir, 'km.log'), maxBytes=1024*1024, backupCount=10)
log_handler_pm.setLevel(log_level)
log_handler_pm.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
app.logger.addHandler(log_handler_pm)

# Define a directory to store private keys
KEY_DIRECTORY = "keys"

@app.route("/get-public-key", methods=["GET", "POST"])
def get_public_key():
    res = request.get_json()
    votee = res["votee"]
    sk = res['sk'].encode('cp437')

    performance_dict = {}
    performance_dict['eval'] = res['eval']
    performance_dict['iteration'] = res['iteration']
    performance_dict['route'] = '/get-public-key'

    # Check if the file exists
    file_path = os.path.join(eval_dir, res['eval'], 'key_manager_' + str(res['eval']) + '.csv')
    file_exists = os.path.isfile(file_path)

    t1 = time.perf_counter()

    # Create a subdirectory within the "keys" directory if it doesn't exist
    subdirectory_path = os.path.join(KEY_DIRECTORY, str(votee))

    if not os.path.exists(subdirectory_path):
        os.makedirs(subdirectory_path)

    t3 = time.perf_counter()
    tracemalloc.start()
    start_memory = tracemalloc.get_traced_memory()[0]
    # Generate a new RSA private key
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
        backend=default_backend()
    )
    end_memory = tracemalloc.get_traced_memory()[0]
    tracemalloc.stop()
    memory_usage_diff = end_memory - start_memory
    performance_dict['size_rsa_sk'] = memory_usage_diff
    t4 = time.perf_counter()
    performance_dict['time_gen_rsa_sk'] = round(t4 - t3, 2)

    # Create a directory to store the private key if it doesn't exist
    if not os.path.exists(KEY_DIRECTORY):
        os.makedirs(KEY_DIRECTORY)

    # Serialize and store the private key as a file
    private_key_file = os.path.join(subdirectory_path, "private_key.pem")  # Store in the subdirectory
    with open(private_key_file, "wb") as key_file:
        key_file.write(private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        ))

    public_key = private_key.public_key()

    t5 = time.perf_counter()
    # Encrypt the file with Fernet
    fernet_key = Fernet.generate_key()
    fernet_cipher = Fernet(fernet_key)

    tracemalloc.start()
    start_memory = tracemalloc.get_traced_memory()[0]
    fernet_encrypted_sk = fernet_cipher.encrypt(sk)
    end_memory = tracemalloc.get_traced_memory()[0]
    tracemalloc.stop()
    memory_usage_diff = end_memory - start_memory
    performance_dict['size_enc_sk1_fernet'] = memory_usage_diff
    t6 = time.perf_counter()
    performance_dict['time_enc_sk1_fernet'] = round(t6 - t5, 2)

    t7 = time.perf_counter()
    tracemalloc.start()
    start_memory = tracemalloc.get_traced_memory()[0]
    # Encrypt the Fernet key with RSA
    encrypted_fernet_key = public_key.encrypt(
        fernet_key,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )
    end_memory = tracemalloc.get_traced_memory()[0]
    tracemalloc.stop()
    memory_usage_diff = end_memory - start_memory
    performance_dict['size_enc_fernet_with_rsa'] = memory_usage_diff
    t8 = time.perf_counter()
    performance_dict['time_enc_fernet_with_rsa'] = round(t8 - t7, 2)

    t2 = time.perf_counter()
    performance_dict['time'] = round(t2 - t1, 2)

    known_columns = ['eval', 'iteration', 'route', 'time', 'size_rsa_sk', 'time_gen_rsa_sk', 'size_enc_sk1_fernet', 'time_enc_sk1_fernet', 'size_enc_fernet_with_rsa', 'time_enc_fernet_with_rsa']
    df_iteration = pd.DataFrame(columns=known_columns)
    data_dict = {col: [performance_dict.get(col, None)] for col in known_columns}
    df_to_append = pd.DataFrame(data_dict)
    df_iteration = pd.concat([df_iteration, df_to_append], ignore_index=True, sort=False)

    df_iteration.to_csv(file_path, mode='a', header=not file_exists, index=False)

    # Serialize the public key as a PEM-encoded string
    """public_key_pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    )"""

    return jsonify({'enc_fernet_key': base64.b64encode(encrypted_fernet_key).decode('utf-8'),
                    'fernet_enc_sk': base64.b64encode(fernet_encrypted_sk).decode('utf-8')})


@app.route("/get-secret-key", methods=["GET","POST"])
def get_secret_key():
    res = request.get_json()
    votee = res["inquired"]

    performance_dict = {}
    performance_dict['eval'] = res['eval']
    performance_dict['iteration'] = res['iteration']
    performance_dict['route'] = '/get-secret-key'

    # Check if the file exists
    file_path = os.path.join(eval_dir, res['eval'], 'key_manager_' + str(res['eval']) + '.csv')
    file_exists = os.path.isfile(file_path)

    t1 = time.perf_counter()

    # Load the private key from the file associated with the votee
    private_key_file = os.path.join(KEY_DIRECTORY, str(votee), "private_key.pem")

    if not os.path.exists(private_key_file):
        # Handle the case where the private key file doesn't exist for the specified votee
        return jsonify({"error": "Private key not found for this votee."}), 404

    with open(private_key_file, "rb") as key_file:
        private_key = serialization.load_pem_private_key(
            key_file.read(),
            password=None,
            backend=default_backend()
        )

    # Serialize the private key as a PEM-encoded string
    private_key_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
    )

    t2 = time.perf_counter()
    performance_dict['time'] = round(t2 - t1, 2)

    known_columns = ['eval', 'iteration', 'route', 'time']
    df_iteration = pd.DataFrame(columns=known_columns)
    data_dict = {col: [performance_dict.get(col, None)] for col in known_columns}
    df_to_append = pd.DataFrame(data_dict)
    df_iteration = pd.concat([df_iteration, df_to_append], ignore_index=True, sort=False)

    df_iteration.to_csv(file_path, mode='a', header=not file_exists, index=False)

    return jsonify({"private_key": base64.b64encode(private_key_pem).decode("utf-8")})

if __name__ == '__main__':
   app.run(debug=config["DEBUG"]["flask_debug"], port=config["KeyManager"]["flask_port"])