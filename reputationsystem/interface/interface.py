import requests
import tomli
import os
import logging
from logging.handlers import RotatingFileHandler

with open("./config.toml", mode="rb") as fp:
    config = tomli.load(fp)

# Define the log file directory path (parent_directory/logging)
log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logging')

# Ensure the log directory exists, create it if it doesn't
os.makedirs(log_dir, exist_ok=True)

# Get the log level from the configuration
log_level = config["Logging"]["log_level"]

# Logging configuration
log_handler = RotatingFileHandler(os.path.join(log_dir, 'interface.log'), maxBytes=1024*1024, backupCount=10)
log_handler.setLevel(log_level)
log_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))

# Create a logger and add the log handler to it
logger = logging.getLogger('my_logger')
logger.setLevel(log_level)
logger.addHandler(log_handler)

#PSEUDONYM MANAGER INTERFACE
def request_recentpseudonym(systemid, eval, iteration):
    data = {'systemid': systemid, 'eval' : eval, 'iteration': iteration}
    response = requests.post(url="http://127.0.0.1:"+str(config["PseudonymManager"]["flask_port"])+"/recent-pseudonym", json=data)
    return response

def request_generalpseudonym(systemid, eval, iteration):
    data = {'systemid': systemid, 'eval' : eval, 'iteration': iteration}
    response = requests.post(url="http://127.0.0.1:"+str(config["PseudonymManager"]["flask_port"])+"/general-pseudonym", json=data)
    return response

#REPUTATION MANAGER INTERFACE
def request_rrecent(pseudonym, votee, eval, iteration):
    data = {'pseudonym': pseudonym, 'user': votee, 'eval' : eval, 'iteration': iteration}
    response = requests.post(url="http://127.0.0.1:"+str(config["ReputationManager"]["flask_port"])+"/rrecent", json=data)
    return response

def request_rgeneral(pseudonym, inquired, eval, iteration):
    data = {'pseudonym': pseudonym, 'inquired': inquired, 'eval' : eval, 'iteration': iteration}
    response = requests.post(url="http://127.0.0.1:"+str(config["ReputationManager"]["flask_port"])+"/rgeneral", json=data)
    return response

def submit_rating(pseudonym, votee, rating, eval, iteration):
    data = {'pseudonym': pseudonym, 'votee': votee, 'rating': rating, 'eval' : eval, 'iteration': iteration}
    response = requests.post(url="http://127.0.0.1:"+str(config["ReputationManager"]["flask_port"])+"/submit-rating", json=data)
    return response

#REPUTATION ENGINE INTERFACE
def register_vote(voterid, voteeid, rating_num, rating_fields, rating_types, rating_limits, eval, iteration):
    data = {'voterid': voterid, 'voteeid': voteeid, 'rating_num': rating_num, 'rating_fields': rating_fields, 'rating_types': rating_types, 'rating_limits': rating_limits, 'eval' : eval, 'iteration': iteration}
    response = requests.post(url="http://127.0.0.1:"+str(config["ReputationEngine"]["flask_port"])+"/register-vote", json=data)
    return response

def send_ratinginfo(voterid, voteeid, subrating, subcount, rrecent, pseudonym, voteid, enc_json, sign_rating, sign_count, eval, iteration):
    data = {'voterid': voterid, 'voteeid': voteeid, 'subrating': subrating, 'subcount': subcount, 'rrecent': rrecent,
            'pseudonym': pseudonym, "voteid": voteid, "enc_json": enc_json, 'signature_rating': sign_rating, 'signature_count': sign_count, 'eval' : eval, 'iteration': iteration}
    response = requests.post(url="http://127.0.0.1:"+str(config["ReputationEngine"]["flask_port"])+"/send-ratinginfo", json=data)
    return response

#VOTEE INTERFACE
def request_objdata(reputation_engine_id, voterid, measure, eval, iteration):
    data = {'reputation_engine_id': reputation_engine_id, 'voter': voterid,
                  'measure': measure, 'eval' : eval, 'iteration': iteration}
    response = requests.post(url="http://127.0.0.1:"+str(config["Votee"]["flask_port"])+"/obj-data", json=data)
    return response

def send_secret_key(votee, sk, context, pk, eval, iteration):
    data = {'votee': votee, 'secret_key': sk, 'context': context, 'pk': pk, 'eval' : eval, 'iteration': iteration}
    response = requests.post(url="http://127.0.0.1:" + str(config["Votee"]["flask_port"]) + "/store-secret-key", json=data)
    return response

def get_secret_key(votee, eval, iteration):
    data = {'votee': votee, 'eval' : eval, 'iteration': iteration}
    response = requests.post(url="http://127.0.0.1:" + str(config["Votee"]["flask_port"]) + "/get-secret-key", json=data)
    return response

#VERIFICATION ENGINE INTERFACE
def send_for_verification_no_enc(pseudonym, eq_val, sign, enc_json, eval, iteration):
    data = {'pseudonym': pseudonym, 'eq_val': eq_val, 'sign': sign, 'enc_json': enc_json, 'eval' : eval, 'iteration': iteration}
    response = requests.post(url="http://127.0.0.1:"+str(config["VerificationEngine"]["flask_port"])+"/verify-rating-no-enc", json=data)
    return response

def send_for_verification(pseudonym, r_recent, r_sum_signs, f_sum_signs, votee_enc_json, voter_enc_json, eval, iteration):
    data = {'pseudonym': pseudonym, 'r_recent': r_recent, 'r_sum_signs': r_sum_signs, 'f_sum_signs': f_sum_signs, 'votee_enc_json': votee_enc_json, 'voter_enc_json': voter_enc_json, 'eval' : eval, 'iteration': iteration}
    response = requests.post(url="http://127.0.0.1:"+str(config["VerificationEngine"]["flask_port"])+"/verify-rating", json=data)
    return response

#KEY MANAGER
def request_pk(pseudonym, votee, eval, iteration):
    data = {'pseudonym': pseudonym, 'votee': votee, 'eval' : eval, 'iteration': iteration}
    response = requests.post(url="http://127.0.0.1:"+str(config["KeyManager"]["flask_port"])+"/get-public-key", json=data)
    return response

def request_sk(pseudonym, inquired, eval, iteration):
    data = {'pseudonym': pseudonym, 'inquired': inquired, 'eval' : eval, 'iteration': iteration}
    response = requests.post(url="http://127.0.0.1:" + str(config["KeyManager"]["flask_port"]) + "/get-secret-key",
                             json=data)
    return response