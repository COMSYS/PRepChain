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

from interface.interface import request_recentpseudonym, request_generalpseudonym, request_rrecent, register_vote, send_ratinginfo, send_for_verification_no_enc, send_for_verification, send_secret_key, get_secret_key, request_pk, request_sk

with open("config.toml", mode="rb") as fp:
    config = tomli.load(fp)

# Define the log file directory path (parent_directory/logging)
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logging')

# Ensure the log directory exists, create it if it doesn't
os.makedirs(log_dir, exist_ok=True)

# Get the log level from the configuration
log_level = config["Logging"]["log_level"]

# Logging configuration
log_handler = RotatingFileHandler(os.path.join(log_dir, 'rating.log'), maxBytes=1024*1024, backupCount=10)
log_handler.setLevel(log_level)
log_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))

# Create a logger and add the log handler to it
logger = logging.getLogger('my_logger')
logger.setLevel(log_level)
logger.addHandler(log_handler)

def generate_inputdata(x,y,z,limit):
    # Randomly select x elements from sub_fields while preserving order
    # selected_sub = random.sample(config['Rating']['sub_fields'], x) #Sample changes order in sublist, breaks vector
    # Ensure x is not greater than the length of sub_fields
    x = min(x, len(config['Rating']['sub_fields']))

    # Randomly select x indices
    selected_indices = random.sample(range(len(config['Rating']['sub_fields'])), x)

    # Create a list of selected values while preserving their order
    selected_sub = [config['Rating']['sub_fields'][i] for i in sorted(selected_indices)]

    # Randomly select y elements from obj_fields while preserving order
    #selected_obj = random.sample(config['Rating']['obj_fields'], y)
    y = min(y, len(config['Rating']['obj_fields']))

    # Randomly select x indices
    selected_indices = random.sample(range(len(config['Rating']['obj_fields'])), y)

    # Create a list of selected values while preserving their order
    selected_obj = [config['Rating']['obj_fields'][i] for i in sorted(selected_indices)]

    # Randomly select z elements from tru_fields while preserving order
    #selected_tru = random.sample(config['Rating']['tru_fields'], z)
    z = min(z, len(config['Rating']['tru_fields']))

    # Randomly select x indices
    selected_indices = random.sample(range(len(config['Rating']['tru_fields'])), z)

    # Create a list of selected values while preserving their order
    selected_tru = [config['Rating']['tru_fields'][i] for i in sorted(selected_indices)]

    complete_selected_ratings = selected_sub + selected_obj + selected_tru

    # Create lists of associated values for selected_obj and selected_tru
    obj_types = [config['Evaldata']['type_'+ str(obj)] for obj in selected_obj]
    tru_types = [config['Evaldata']['type_'+ str(tru)] for tru in selected_tru]
    complete_types = obj_types + tru_types

    # Create a list with all values set to the limit
    limits = [limit] * len(complete_types)

    # Create a list of random values between 1 and 10 based on the length of selected_sub
    random_sub_vote = [random.randint(config['Rating']['sub_lowerbound'], config['Rating']['sub_upperbound']) for _ in range(len(selected_sub))]

    return complete_selected_ratings, complete_types, limits, random_sub_vote

#@profile
def main(iteration, eval_run):
    t0 = time.perf_counter(), time.process_time()
    performance_dict = {}
    performance_dict['eval'] = eval_run
    performance_dict['iteration'] = [iteration]

    x = random.randint(1, config['Rating']['sub_num'])
    y = random.randint(0, config['Rating']['obj_num'])
    z = random.randint(0, config['Rating']['tru_num'])
    complete_selected_ratings, complete_types, limits, random_sub_vote = generate_inputdata(x, y, z, 10)
    participants = [2001, 2002]
    # Randomly select one participant as the voter
    voter = random.choice(participants)

    # Assign the other participant as the votee
    votee = participants[0] if voter == participants[1] else participants[1]

    print(x,y,z,complete_selected_ratings, complete_types, limits, random_sub_vote)

    parser = argparse.ArgumentParser(
        prog="Rating Process",
        description="A script to simulate the rating process in our reputation system")
    parser.add_argument(
        "-bid",
        "--BusinessID",
        type=int,
        default=1,
        help="insert business ID here (default: %(default)s)")
    parser.add_argument(
        "-sid",
        "--voterSystemID",
        type=int,
        default=voter,
        help="insert system ID here (default: %(default)s)")
    parser.add_argument(
        "-v",
        "--voteeSystemID",
        type=int,
        default=votee,
        help="insert system ID of votee here (default: %(default)s)", )
    parser.add_argument(
        "-num",
        "--ratingNumbers",
        nargs=3,
        type=int,
        default=[x,y,z],
        help="insert the number of features in the rating here in the form #s #o #t (default: %(default)s)")
    parser.add_argument(
        "-f",
        "--ratingFeatures",
        nargs="+",
        #default=["service", "quality", "goodsreceipt", "temperatursaegeblatt", "humidity", "trustedgoodsreceipt", "trustedtemperature", "trustedhumidity"],
        default=complete_selected_ratings,
        help="insert the features in the rating here in the form s_i s_i+1 ... o_j o_j+1 ... t_k t_k+1 ... (default: %(default)s)")
    parser.add_argument(
        "-t",
        "--ratingTypes",
        nargs="+",
        default=complete_types,
        help="insert the type of features in the objective part of the rating here in the form 'diff' [o1_min, o1_max] [o2_min, o2_max] [t1_min, t1_max] [t2_min, t2_max] (default: %(default)s)")
    parser.add_argument(
        "-l",
        "--ratingLimits",
        nargs="+",
        default=limits,
        help="insert the limit in percent for each of the objective rating types in the form limit(o1) limit(o2) limit(t1) limit(t2) (default: %(default)s)")
    parser.add_argument(
        "-r",
        "--rating",
        nargs=2,
        type=int,
        default=random_sub_vote,
        help="insert the values for the subjective rating here (default: %(default)s)")

    args = parser.parse_args()
    s_num, o_num, t_num = args.ratingNumbers

    #performance_dict['input'] = [args.voterSystemID, args.voteeSystemID, x, y, z, complete_selected_ratings, complete_types, limits]
    performance_dict['voter'] = args.voterSystemID
    performance_dict['votee'] = args.voteeSystemID
    performance_dict['sub_num'] = x
    performance_dict['obj_num'] = y
    performance_dict['tru_num'] = z
    performance_dict['sub_rating'] = str(args.rating)
    performance_dict['rating_features'] = str(complete_selected_ratings)
    performance_dict['rating_types'] = str(complete_types)
    performance_dict['rating_limits'] = str(limits)

    if s_num + o_num + t_num != len(args.ratingFeatures):
        print("Error: Number of rating features does not match sum of rating numbers.")
        return

    if o_num + t_num != len(args.ratingTypes) and o_num + t_num != len(args.ratingLimits):
        print("Error: Number of rating types and/or number of rating limits does not match objective and trusted objective rating number.")
        return

    if s_num != len(args.rating):
        print("Error: Dimension of subjective rating does not match number of subjective ratings.")
        return

    db_ratingid = register_vote(args.voterSystemID, args.voteeSystemID, args.ratingNumbers, args.ratingFeatures, args.ratingTypes, args.ratingLimits, eval_run, iteration).json()
    if "voteid" not in db_ratingid.keys():
        print("Error: Could not register rating, please try again.")
    else:
        print(("Successfully registered vote: {}").format(db_ratingid["voteid"]))

    print("Request Pseudonym to query own Reputation")
    t1 = time.perf_counter(), time.process_time()
    voter_pseudonym_recent = request_recentpseudonym(args.BusinessID, eval_run, iteration).json()
    t2 = time.perf_counter(), time.process_time()
    performance_dict['voter_pseudonym_recent_perf'] = round(t2[0]-t1[0],2)
    #performance_dict['voter_pseudonym_recent_proc'] = round(t2[1]-t1[1],2)

    print("Request Pseudonym to query votee Reputation/Public Key")
    t1 = time.perf_counter(), time.process_time()
    votee_pseudonym_recent = request_recentpseudonym(args.BusinessID, eval_run, iteration).json()
    t2 = time.perf_counter(), time.process_time()
    performance_dict['votee_pseudonym_recent_perf'] = round(t2[0] - t1[0], 2)
    #performance_dict['votee_pseudonym_recent_proc'] = round(t2[1] - t1[1], 2)

    print("Request Pseudonym to verify rating")
    t1 = time.perf_counter(), time.process_time()
    verification_pseudonym_recent = request_recentpseudonym(args.BusinessID, eval_run, iteration).json()
    t2 = time.perf_counter(), time.process_time()
    performance_dict['verification_pseudonym_recent_perf'] = round(t2[0] - t1[0], 2)
    #performance_dict['verification_pseudonym_recent_proc'] = round(t2[1] - t1[1], 2)

    #print(f"Pseudonym for voter to request own reputation : {voter_pseudonym_recent}")
    #print(f"Pseudonym for voter to request votee reputation : {votee_pseudonym_recent}")
    #print(f"Pseudonym for voter to verify rating : {verification_pseudonym_recent}")

    #print("Request Votee R_Recent")
    #response = request_rrecent(pseudonym_request["recentpseudonym"], args.voteeSystemID)

    print("Request Voter Aggregated R_Recent")
    #response = request_rrecent(pseudonym_request["recentpseudonym"], args.voterSystemID)
    t1 = time.perf_counter(), time.process_time()
    voter_reputation_request = request_rrecent(voter_pseudonym_recent["recentpseudonym"], args.voterSystemID, eval_run, iteration).json()
    t2 = time.perf_counter(), time.process_time()
    performance_dict['voter_reputation_request_perf'] = round(t2[0] - t1[0], 2)
    #performance_dict['voter_reputation_request_proc'] = round(t2[1] - t1[1], 2)

    #print(f"Response to request of voter aggregated R_Recent : {voter_reputation_request}")
    print("Request Votee Aggregated R_Recent/Public Key")
    t1 = time.perf_counter(), time.process_time()
    votee_reputation_request = request_rrecent(votee_pseudonym_recent["recentpseudonym"], args.voteeSystemID, eval_run, iteration).json()
    t2 = time.perf_counter(), time.process_time()
    performance_dict['votee_reputation_request_perf'] = round(t2[0] - t1[0], 2)
    #performance_dict['votee_reputation_request_proc'] = round(t2[1] - t1[1], 2)
    #print(f"Response to request of votee aggregated R_Recent : {votee_reputation_request}")

    #UNTERSCHEIDUNG Votee keine Reputation -> Public Key usw. initialisieren
    if votee_reputation_request["total_count"] == 0:
        print(
            f'Votee {votee_reputation_request["total_count"]} Rep, Voter {voter_reputation_request["total_count"]} Rep')

        print(f"[Client] Initializing Pyfhel session and data...")
        HE_votee = Pyfhel(context_params={'scheme': 'ckks', 'n': 2 ** 13, 'scale': 2 ** 30, 'qi_sizes': [30] * 5})
        t1 = time.perf_counter(), time.process_time()
        HE_votee.keyGen()  # Generates both a public and a private key
        t2 = time.perf_counter(), time.process_time()
        performance_dict['HE_votee.keyGen()_request_perf'] = round(t2[0] - t1[0], 2)
        #performance_dict['HE_votee.keyGen()_request_proc'] = round(t2[1] - t1[1], 2)

        t1 = time.perf_counter(), time.process_time()
        HE_votee.relinKeyGen()
        t2 = time.perf_counter(), time.process_time()
        performance_dict['HE_votee.relinKeyGen()_perf'] = round(t2[0] - t1[0], 2)
        #performance_dict['HE_votee.relinKeyGen()_proc'] = round(t2[1] - t1[1], 2)

        t1 = time.perf_counter(), time.process_time()
        HE_votee.rotateKeyGen()
        t2 = time.perf_counter(), time.process_time()
        performance_dict['HE_votee.rotateKeyGen()_perf'] = round(t2[0] - t1[0], 2)
        #performance_dict['HE_votee.rotateKeyGen()_proc'] = round(t2[1] - t1[1], 2)
        #UNTERSCHEIDUNG Voter und Votee keine Reputation -> Äquivalenzklasse wird unencrypted übertragen
        if voter_reputation_request["total_count"] == 0:
            performance_dict['VoteeRep'] = votee_reputation_request["total_count"]
            performance_dict['VoterRep'] = voter_reputation_request["total_count"]

            # VE0,VR0
            eq_class1 = config["Rating"]["eq_classes"][0]
            eq_class2 = config["Rating"]["eq_classes"][1]
            eq_class3 = config["Rating"]["eq_classes"][2]
            if eq_class1[0] < int.from_bytes(voter_reputation_request["R_recent"].encode('cp437'), 'little') <= eq_class1[
                1]:
                eq_class = 0
            elif eq_class2[0] < int.from_bytes(voter_reputation_request["R_recent"].encode('cp437'), 'little') <= eq_class2[
                1]:
                eq_class = 1
            elif eq_class3[0] < int.from_bytes(voter_reputation_request["R_recent"].encode('cp437'), 'little') <= \
                    eq_class3[1]:
                eq_class = 2
            else:
                return print('ERROR: Can not assign correct eq_class')

            t1 = time.perf_counter(), time.process_time()
            padded_rating, padded_count = compute_padded_rating(args.rating, args.ratingNumbers, args.ratingFeatures,
                                                                eq_class)
            t2 = time.perf_counter(), time.process_time()
            performance_dict['padded_rating_perf'] = round(t2[0] - t1[0], 2)
            #performance_dict['padded_rating_proc'] = round(t2[1] - t1[1], 2)

            #print(padded_rating, padded_count)

            plain_rating = np.array(padded_rating)
            plain_count = np.array(padded_count)

            t1 = time.perf_counter(), time.process_time()
            enc_rating = HE_votee.encrypt(plain_rating)
            t2 = time.perf_counter(), time.process_time()
            performance_dict['enc_rating_perf'] = round(t2[0] - t1[0], 2)
            #performance_dict['enc_rating_proc'] = round(t2[1] - t1[1], 2)

            t1 = time.perf_counter(), time.process_time()
            enc_count = HE_votee.encrypt(plain_count)
            t2 = time.perf_counter(), time.process_time()
            performance_dict['enc_count_perf'] = round(t2[0] - t1[0], 2)
            #performance_dict['enc_count_proc'] = round(t2[1] - t1[1], 2)

            tracemalloc.start()
            start_memory = tracemalloc.get_traced_memory()[0]
            s_context = HE_votee.to_bytes_context()
            end_memory = tracemalloc.get_traced_memory()[0]
            tracemalloc.stop()
            memory_usage_diff = end_memory - start_memory
            performance_dict['size_votee_s_context'] = memory_usage_diff

            tracemalloc.start()
            start_memory = tracemalloc.get_traced_memory()[0]
            s_public_key = HE_votee.to_bytes_public_key()
            end_memory = tracemalloc.get_traced_memory()[0]
            tracemalloc.stop()
            memory_usage_diff = end_memory - start_memory
            performance_dict['size_votee_s_public_key'] = memory_usage_diff

            tracemalloc.start()
            start_memory = tracemalloc.get_traced_memory()[0]
            s_secret_key = HE_votee.to_bytes_secret_key()  # FOR CHECKING IF ENC AND PLAIN CORRECT
            end_memory = tracemalloc.get_traced_memory()[0]
            tracemalloc.stop()
            memory_usage_diff = end_memory - start_memory
            performance_dict['size_votee_s_secret_key'] = memory_usage_diff

            tracemalloc.start()
            start_memory = tracemalloc.get_traced_memory()[0]
            s_relin_key = HE_votee.to_bytes_relin_key()
            end_memory = tracemalloc.get_traced_memory()[0]
            tracemalloc.stop()
            memory_usage_diff = end_memory - start_memory
            performance_dict['size_votee_s_relin_key'] = memory_usage_diff

            tracemalloc.start()
            start_memory = tracemalloc.get_traced_memory()[0]
            s_rotate_key = HE_votee.to_bytes_rotate_key()
            end_memory = tracemalloc.get_traced_memory()[0]
            tracemalloc.stop()
            memory_usage_diff = end_memory - start_memory
            performance_dict['size_votee_s_rotate_key'] = memory_usage_diff

            tracemalloc.start()
            start_memory = tracemalloc.get_traced_memory()[0]
            s_enc_rating = enc_rating.to_bytes()
            end_memory = tracemalloc.get_traced_memory()[0]
            tracemalloc.stop()
            memory_usage_diff = end_memory - start_memory
            performance_dict['size_s_enc_rating'] = memory_usage_diff

            tracemalloc.start()
            start_memory = tracemalloc.get_traced_memory()[0]
            s_enc_count = enc_count.to_bytes()
            end_memory = tracemalloc.get_traced_memory()[0]
            tracemalloc.stop()
            memory_usage_diff = end_memory - start_memory
            performance_dict['size_s_enc_count'] = memory_usage_diff

            #print(f"[Client] sending HE_client={HE_votee} and enc_rating={enc_rating} and enc_count={enc_count}")
            enc_json = {
                'context': s_context.decode('cp437'),
                'pk': s_public_key.decode('cp437'),
                'sk': s_secret_key.decode('cp437'),
                'rlk': s_relin_key.decode('cp437'),
                'rtk': s_rotate_key.decode('cp437'),
                's_enc_rating': s_enc_rating.decode('cp437'),
                's_enc_count': s_enc_count.decode('cp437'),
            }

            print("Send Rating and signed unencrypted voter level to Verification Engine")
            t1 = time.perf_counter(), time.process_time()
            verification_request = send_for_verification_no_enc(verification_pseudonym_recent["recentpseudonym"],
                                                                voter_reputation_request["R_recent"],
                                                                voter_reputation_request["signature"], enc_json, eval_run, iteration).json()
            t2 = time.perf_counter(), time.process_time()
            performance_dict['verification_request_perf'] = round(t2[0] - t1[0], 2)
            #performance_dict['verification_request_proc'] = round(t2[1] - t1[1], 2)

            #print(verification_request)

            enc_json.pop('sk') #Remove secret key from data sent to RE

            print("Send Rating to Reputation Engine")
            t1 = time.perf_counter(), time.process_time()
            rating_request = send_ratinginfo(args.voterSystemID, args.voteeSystemID, padded_rating, padded_count,
                                             voter_reputation_request["R_recent"],
                                             voter_pseudonym_recent["recentpseudonym"],
                                             db_ratingid["voteid"], enc_json, verification_request['signature_rating'],
                                             verification_request['signature_count'], eval_run, iteration).json()
            t2 = time.perf_counter(), time.process_time()
            performance_dict['rating_request_perf'] = round(t2[0] - t1[0], 2)
            #performance_dict['rating_request_proc'] = round(t2[1] - t1[1], 2)

            print("Send Secret Key to Votee")
            t1 = time.perf_counter(), time.process_time()
            sk_send = send_secret_key(args.voteeSystemID, HE_votee.to_bytes_secret_key().decode('cp437'), HE_votee.to_bytes_context().decode('cp437'), HE_votee.to_bytes_public_key().decode('cp437'), eval_run, iteration)
            t2 = time.perf_counter(), time.process_time()
            performance_dict['sk_send_perf'] = round(t2[0] - t1[0], 2)
            #performance_dict['sk_send_proc'] = round(t2[1] - t1[1], 2)

        else: #UNTERSCHEIDUNG Votee keine Reputation Voter hat Reputation -> Äquivalenzklasse berechnen
            #VE0,VR1
            performance_dict['VoteeRep'] = votee_reputation_request["total_count"]
            performance_dict['VoterRep'] = voter_reputation_request["total_count"]

            t1 = time.perf_counter(), time.process_time()
            voter_enc_keys = voter_reputation_request["enc_keys"]
            t2 = time.perf_counter(), time.process_time()
            performance_dict['voter_enc_keys_perf'] = round(t2[0] - t1[0], 2)
            #performance_dict['voter_enc_keys_proc'] = round(t2[1] - t1[1], 2)

            key_dir = get_keydirectory_path(str(args.voterSystemID))
            HE_voter = Pyfhel()

            HE_voter.load_context(key_dir + "/context") #Context and Secret Key stored at Voter
            HE_voter.load_secret_key(key_dir + "/sec.key")
            HE_voter.from_bytes_public_key(voter_enc_keys['pk'].encode('cp437'))
            HE_voter.from_bytes_relin_key(voter_enc_keys['rlk'].encode('cp437'))
            HE_voter.from_bytes_rotate_key(voter_enc_keys['rtk'].encode('cp437'))

            t1 = time.perf_counter(), time.process_time()
            avg_aggr_rep_lvl = compute_rep_level(HE_voter, voter_reputation_request["R_recent"], voter_reputation_request["plain_r_recent"])
            t2 = time.perf_counter(), time.process_time()
            performance_dict['avg_aggr_rep_lvl_perf'] = round(t2[0] - t1[0], 2)
            #performance_dict['avg_aggr_rep_lvl_proc'] = round(t2[1] - t1[1], 2)

            eq_class1 = config["Rating"]["eq_classes"][0]
            eq_class2 = config["Rating"]["eq_classes"][1]
            eq_class3 = config["Rating"]["eq_classes"][2]
            if eq_class1[0] < avg_aggr_rep_lvl <= \
                    eq_class1[
                        1]:
                eq_class = 0
            elif eq_class2[0] < avg_aggr_rep_lvl <= \
                    eq_class2[
                        1]:
                eq_class = 1
            elif eq_class3[0] < avg_aggr_rep_lvl <= \
                    eq_class3[1]:
                eq_class = 2
            else:
                return print('ERROR: Can not assign correct eq_class')

            t1 = time.perf_counter(), time.process_time()
            padded_rating, padded_count = compute_padded_rating(args.rating, args.ratingNumbers, args.ratingFeatures,
                                                                eq_class)
            t2 = time.perf_counter(), time.process_time()
            performance_dict['padded_rating_perf'] = round(t2[0] - t1[0], 2)
            #performance_dict['padded_rating_proc'] = round(t2[1] - t1[1], 2)
            #print(padded_rating, padded_count)

            plain_rating = np.array(padded_rating)
            plain_count = np.array(padded_count)

            t1 = time.perf_counter(), time.process_time()
            enc_rating = HE_votee.encrypt(plain_rating)
            t2 = time.perf_counter(), time.process_time()
            performance_dict['enc_rating_perf'] = round(t2[0] - t1[0], 2)
            #performance_dict['enc_rating_proc'] = round(t2[1] - t1[1], 2)

            t1 = time.perf_counter(), time.process_time()
            enc_count = HE_votee.encrypt(plain_count)
            t2 = time.perf_counter(), time.process_time()
            performance_dict['enc_count_perf'] = round(t2[0] - t1[0], 2)
            #performance_dict['enc_count_proc'] = round(t2[1] - t1[1], 2)

            tracemalloc.start()
            start_memory = tracemalloc.get_traced_memory()[0]
            votee_s_context = HE_votee.to_bytes_context()
            end_memory = tracemalloc.get_traced_memory()[0]
            tracemalloc.stop()
            memory_usage_diff = end_memory - start_memory
            performance_dict['size_votee_s_context'] = memory_usage_diff

            tracemalloc.start()
            start_memory = tracemalloc.get_traced_memory()[0]
            votee_s_public_key = HE_votee.to_bytes_public_key()
            end_memory = tracemalloc.get_traced_memory()[0]
            tracemalloc.stop()
            memory_usage_diff = end_memory - start_memory
            performance_dict['size_votee_s_public_key'] = memory_usage_diff

            tracemalloc.start()
            start_memory = tracemalloc.get_traced_memory()[0]
            votee_s_secret_key = HE_votee.to_bytes_secret_key()  # FOR CHECKING IF ENC AND PLAIN CORRECT
            end_memory = tracemalloc.get_traced_memory()[0]
            tracemalloc.stop()
            memory_usage_diff = end_memory - start_memory
            performance_dict['size_votee_s_secret_key'] = memory_usage_diff

            tracemalloc.start()
            start_memory = tracemalloc.get_traced_memory()[0]
            votee_s_relin_key = HE_votee.to_bytes_relin_key()
            end_memory = tracemalloc.get_traced_memory()[0]
            tracemalloc.stop()
            memory_usage_diff = end_memory - start_memory
            performance_dict['size_votee_s_relin_key'] = memory_usage_diff

            tracemalloc.start()
            start_memory = tracemalloc.get_traced_memory()[0]
            votee_s_rotate_key = HE_votee.to_bytes_rotate_key()
            end_memory = tracemalloc.get_traced_memory()[0]
            tracemalloc.stop()
            memory_usage_diff = end_memory - start_memory
            performance_dict['size_votee_s_rotate_key'] = memory_usage_diff

            tracemalloc.start()
            start_memory = tracemalloc.get_traced_memory()[0]
            s_enc_rating = enc_rating.to_bytes()
            end_memory = tracemalloc.get_traced_memory()[0]
            tracemalloc.stop()
            memory_usage_diff = end_memory - start_memory
            performance_dict['size_s_enc_rating'] = memory_usage_diff

            tracemalloc.start()
            start_memory = tracemalloc.get_traced_memory()[0]
            s_enc_count = enc_count.to_bytes()
            end_memory = tracemalloc.get_traced_memory()[0]
            tracemalloc.stop()
            memory_usage_diff = end_memory - start_memory
            performance_dict['size_s_enc_count'] = memory_usage_diff

            tracemalloc.start()
            start_memory = tracemalloc.get_traced_memory()[0]
            voter_s_context = HE_voter.to_bytes_context()
            end_memory = tracemalloc.get_traced_memory()[0]
            tracemalloc.stop()
            memory_usage_diff = end_memory - start_memory
            performance_dict['size_voter_s_context'] = memory_usage_diff

            tracemalloc.start()
            start_memory = tracemalloc.get_traced_memory()[0]
            voter_s_public_key = HE_voter.to_bytes_public_key()
            end_memory = tracemalloc.get_traced_memory()[0]
            tracemalloc.stop()
            memory_usage_diff = end_memory - start_memory
            performance_dict['size_voter_s_public_key'] = memory_usage_diff

            tracemalloc.start()
            start_memory = tracemalloc.get_traced_memory()[0]
            voter_s_secret_key = HE_voter.to_bytes_secret_key()  # FOR CHECKING IF ENC AND PLAIN CORRECT
            end_memory = tracemalloc.get_traced_memory()[0]
            tracemalloc.stop()
            memory_usage_diff = end_memory - start_memory
            performance_dict['size_voter_s_secret_key'] = memory_usage_diff

            tracemalloc.start()
            start_memory = tracemalloc.get_traced_memory()[0]
            voter_s_relin_key = HE_voter.to_bytes_relin_key()
            end_memory = tracemalloc.get_traced_memory()[0]
            tracemalloc.stop()
            memory_usage_diff = end_memory - start_memory
            performance_dict['size_voter_s_relin_key'] = memory_usage_diff

            tracemalloc.start()
            start_memory = tracemalloc.get_traced_memory()[0]
            voter_s_rotate_key = HE_voter.to_bytes_rotate_key()
            end_memory = tracemalloc.get_traced_memory()[0]
            tracemalloc.stop()
            memory_usage_diff = end_memory - start_memory
            performance_dict['size_voter_s_rotate_key'] = memory_usage_diff

            votee_enc_json = {
                'context': votee_s_context.decode('cp437'),
                'sk': votee_s_secret_key.decode('cp437'),
                'pk': votee_s_public_key.decode('cp437'),
                'rtk': votee_s_rotate_key.decode('cp437'),
                'rlk': votee_s_relin_key.decode('cp437'),
                's_enc_rating': s_enc_rating.decode('cp437'),
                's_enc_count': s_enc_count.decode('cp437'),
            }

            voter_enc_json = {
                'context': voter_s_context.decode('cp437'),
                'pk': voter_s_public_key.decode('cp437'),
                'sk': voter_s_secret_key.decode('cp437'),
                'rlk': voter_s_relin_key.decode('cp437'),
                'rtk': voter_s_rotate_key.decode('cp437'),
                's_enc_rating': s_enc_rating.decode('cp437'),
                's_enc_count': s_enc_count.decode('cp437'),
            }

            print("Send Rating and signed encrypted voter level to Verification Engine")
            t1 = time.perf_counter(), time.process_time()
            verification_request = send_for_verification(verification_pseudonym_recent["recentpseudonym"],
                                                         voter_reputation_request["R_recent"],
                                                         voter_reputation_request["r_sum_signs"],
                                                         voter_reputation_request["f_sum_signs"], votee_enc_json, voter_enc_json, eval_run, iteration).json()
            t2 = time.perf_counter(), time.process_time()
            performance_dict['verification_request_perf'] = round(t2[0] - t1[0], 2)
            #performance_dict['verification_request_proc'] = round(t2[1] - t1[1], 2)
            #print(verification_request)

            votee_enc_json.pop('sk')  # Remove secret key from data sent to RE

            print("Send Rating to Reputation Engine")
            # print(args.SystemID, args.voteeSystemID, args.rating, reputation_request["R_recent"], pseudonym_request["recentpseudonym"], db_ratingid["voteid"])
            # response = send_ratinginfo(args.SystemID, args.voteeSystemID, padded_rating, padded_count, reputation_request["R_recent"], pseudonym_request["recentpseudonym"], db_ratingid["voteid"])
            t1 = time.perf_counter(), time.process_time()
            rating_request = send_ratinginfo(args.voterSystemID, args.voteeSystemID, padded_rating, padded_count,
                                             voter_reputation_request["R_recent"],
                                             voter_pseudonym_recent["recentpseudonym"],
                                             db_ratingid["voteid"], votee_enc_json, verification_request['signature_rating'],
                                             verification_request['signature_count'], eval_run, iteration).json()
            t2 = time.perf_counter(), time.process_time()
            performance_dict['rating_request_perf'] = round(t2[0] - t1[0], 2)
            #performance_dict['rating_request_proc'] = round(t2[1] - t1[1], 2)

            print("Send Secret Key to Votee")

            t1 = time.perf_counter(), time.process_time()
            sk_send = send_secret_key(args.voteeSystemID, HE_votee.to_bytes_secret_key().decode('cp437'), HE_votee.to_bytes_context().decode('cp437'), HE_votee.to_bytes_public_key().decode('cp437'), eval_run, iteration)
            t2 = time.perf_counter(), time.process_time()
            performance_dict['sk_send_perf'] = round(t2[0] - t1[0], 2)
            #performance_dict['sk_send_proc'] = round(t2[1] - t1[1], 2)

    else: #Votee hat Reputation, Voter nicht
        print(
            f'Votee {votee_reputation_request["total_count"]} Rep, Voter {voter_reputation_request["total_count"]} Rep')

        t1 = time.perf_counter(), time.process_time()
        enc_keys = votee_reputation_request["enc_keys"]
        t2 = time.perf_counter(), time.process_time()
        performance_dict['enc_keys_perf'] = round(t2[0] - t1[0], 2)
        #performance_dict['enc_keys_proc'] = round(t2[1] - t1[1], 2)

        t1 = time.perf_counter(), time.process_time()
        inq_enc_keys = votee_reputation_request["enc_sk"]
        t2 = time.perf_counter(), time.process_time()
        performance_dict['inq_enc_keys_perf'] = round(t2[0] - t1[0], 2)
        #performance_dict['inq_enc_keys_proc'] = round(t2[1] - t1[1], 2)

        print("Request Pseudonym to query decryption from KM")
        t1 = time.perf_counter(), time.process_time()
        km_pseudonym_recent = request_recentpseudonym(args.BusinessID, eval_run, iteration).json()
        t2 = time.perf_counter(), time.process_time()
        performance_dict['km_pseudonym_recent_perf'] = round(t2[0] - t1[0], 2)
        #performance_dict['km_pseudonym_recent_proc'] = round(t2[1] - t1[1], 2)

        t1 = time.perf_counter(), time.process_time()
        enc_sk_request = request_sk(km_pseudonym_recent["recentpseudonym"], args.voteeSystemID, eval_run, iteration).json()
        t2 = time.perf_counter(), time.process_time()
        performance_dict['enc_sk_request_perf'] = round(t2[0] - t1[0], 2)
        #performance_dict['enc_sk_request_proc'] = round(t2[1] - t1[1], 2)

        # Deserialize the private key
        private_key = serialization.load_pem_private_key(
            base64.b64decode(enc_sk_request['private_key']),  # Ensure it's encoded as bytes
            password=None,  # If your private key is password-protected, provide the password here
            backend=default_backend()
        )

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
        #performance_dict['fernet_key_proc'] = round(t2[1] - t1[1], 2)

        t1 = time.perf_counter(), time.process_time()
        fernet_cipher = Fernet(fernet_key)
        t2 = time.perf_counter(), time.process_time()
        performance_dict['fernet_cipher_perf'] = round(t2[0] - t1[0], 2)
        #performance_dict['fernet_cipher_proc'] = round(t2[1] - t1[1], 2)

        t1 = time.perf_counter(), time.process_time()
        decrypted_sk = fernet_cipher.decrypt(base64.b64decode(inq_enc_keys["fernet_enc_sk"]))
        t2 = time.perf_counter(), time.process_time()
        performance_dict['decrypted_sk_perf'] = round(t2[0] - t1[0], 2)
        #performance_dict['decrypted_sk_proc'] = round(t2[1] - t1[1], 2)

        """print("Get Secret Key from Votee")
        sk_get = get_secret_key(args.voteeSystemID).json()"""

        """if sk_get['sk'].encode('cp437') == enc_keys["sk"].encode('cp437'):
            print('SECRET KEYS SAME!')

        else:
            print('SECRET KEYS NOT SAME!')"""


        print(f"[Client] RE-Initializing Votee Pyfhel session and data...")
        HE_votee = Pyfhel()
        HE_votee.from_bytes_context(enc_keys["context"].encode('cp437'))
        HE_votee.from_bytes_public_key(enc_keys["pk"].encode('cp437'))
        HE_votee.from_bytes_relin_key(enc_keys["rlk"].encode('cp437'))
        HE_votee.from_bytes_rotate_key(enc_keys["rtk"].encode('cp437'))
        #HE_votee.from_bytes_secret_key(sk_get['sk'].encode('cp437'))
        HE_votee.from_bytes_secret_key(decrypted_sk)
        #HE_votee.from_bytes_secret_key(enc_keys['sk'].encode('cp437'))

        if voter_reputation_request["total_count"] == 0:
            #VE1,VR0

            performance_dict['VoteeRep'] = votee_reputation_request["total_count"]
            performance_dict['VoterRep'] = voter_reputation_request["total_count"]

            eq_class1 = config["Rating"]["eq_classes"][0]
            eq_class2 = config["Rating"]["eq_classes"][1]
            eq_class3 = config["Rating"]["eq_classes"][2]
            if eq_class1[0] < int.from_bytes(voter_reputation_request["R_recent"].encode('cp437'), 'little') <= \
                    eq_class1[
                        1]:
                eq_class = 0
            elif eq_class2[0] < int.from_bytes(voter_reputation_request["R_recent"].encode('cp437'), 'little') <= \
                    eq_class2[
                        1]:
                eq_class = 1
            elif eq_class3[0] < int.from_bytes(voter_reputation_request["R_recent"].encode('cp437'), 'little') <= \
                    eq_class3[1]:
                eq_class = 2
            else:
                return print('ERROR: Can not assign correct eq_class')

            t1 = time.perf_counter(), time.process_time()
            padded_rating, padded_count = compute_padded_rating(args.rating, args.ratingNumbers, args.ratingFeatures,
                                                                eq_class)
            t2 = time.perf_counter(), time.process_time()
            performance_dict['padded_rating_perf'] = round(t2[0] - t1[0], 2)
            #performance_dict['padded_rating_proc'] = round(t2[1] - t1[1], 2)


            #print(padded_rating, padded_count)

            plain_rating = np.array(padded_rating)
            plain_count = np.array(padded_count)

            t1 = time.perf_counter(), time.process_time()
            enc_rating = HE_votee.encrypt(plain_rating)
            t2 = time.perf_counter(), time.process_time()
            performance_dict['enc_rating_perf'] = round(t2[0] - t1[0], 2)
            #performance_dict['enc_rating_proc'] = round(t2[1] - t1[1], 2)

            t1 = time.perf_counter(), time.process_time()
            enc_count = HE_votee.encrypt(plain_count)
            t2 = time.perf_counter(), time.process_time()
            performance_dict['enc_count_perf'] = round(t2[0] - t1[0], 2)
            #performance_dict['enc_count_proc'] = round(t2[1] - t1[1], 2)

            tracemalloc.start()
            start_memory = tracemalloc.get_traced_memory()[0]
            s_context = HE_votee.to_bytes_context()
            end_memory = tracemalloc.get_traced_memory()[0]
            tracemalloc.stop()
            memory_usage_diff = end_memory - start_memory
            performance_dict['size_votee_s_context'] = memory_usage_diff

            tracemalloc.start()
            start_memory = tracemalloc.get_traced_memory()[0]
            s_public_key = HE_votee.to_bytes_public_key()
            end_memory = tracemalloc.get_traced_memory()[0]
            tracemalloc.stop()
            memory_usage_diff = end_memory - start_memory
            performance_dict['size_votee_s_public_key'] = memory_usage_diff

            tracemalloc.start()
            start_memory = tracemalloc.get_traced_memory()[0]
            s_secret_key = HE_votee.to_bytes_secret_key()  # FOR CHECKING IF ENC AND PLAIN CORRECT
            end_memory = tracemalloc.get_traced_memory()[0]
            tracemalloc.stop()
            memory_usage_diff = end_memory - start_memory
            performance_dict['size_votee_s_secret_key'] = memory_usage_diff

            tracemalloc.start()
            start_memory = tracemalloc.get_traced_memory()[0]
            s_relin_key = HE_votee.to_bytes_relin_key()
            end_memory = tracemalloc.get_traced_memory()[0]
            tracemalloc.stop()
            memory_usage_diff = end_memory - start_memory
            performance_dict['size_votee_s_relin_key'] = memory_usage_diff

            tracemalloc.start()
            start_memory = tracemalloc.get_traced_memory()[0]
            s_rotate_key = HE_votee.to_bytes_rotate_key()
            end_memory = tracemalloc.get_traced_memory()[0]
            tracemalloc.stop()
            memory_usage_diff = end_memory - start_memory
            performance_dict['size_votee_s_rotate_key'] = memory_usage_diff

            tracemalloc.start()
            start_memory = tracemalloc.get_traced_memory()[0]
            s_enc_rating = enc_rating.to_bytes()
            end_memory = tracemalloc.get_traced_memory()[0]
            tracemalloc.stop()
            memory_usage_diff = end_memory - start_memory
            performance_dict['size_s_enc_rating'] = memory_usage_diff

            tracemalloc.start()
            start_memory = tracemalloc.get_traced_memory()[0]
            s_enc_count = enc_count.to_bytes()
            end_memory = tracemalloc.get_traced_memory()[0]
            tracemalloc.stop()
            memory_usage_diff = end_memory - start_memory
            performance_dict['size_s_enc_count'] = memory_usage_diff

            #print(f"[Client] sending HE_client={HE_votee} and enc_rating={enc_rating} and enc_count={enc_count}")
            enc_json = {
                'context': s_context.decode('cp437'),
                'pk': s_public_key.decode('cp437'),
                'sk': s_secret_key.decode('cp437'),
                'rlk': s_relin_key.decode('cp437'),
                'rtk': s_rotate_key.decode('cp437'),
                's_enc_rating': s_enc_rating.decode('cp437'),
                's_enc_count': s_enc_count.decode('cp437'),
            }

            print("Send Rating and signed unencrypted voter level to Verification Engine")
            t1 = time.perf_counter(), time.process_time()
            verification_request = send_for_verification_no_enc(verification_pseudonym_recent["recentpseudonym"],
                                                                voter_reputation_request["R_recent"],
                                                                voter_reputation_request["signature"], enc_json, eval_run, iteration).json()
            t2 = time.perf_counter(), time.process_time()
            performance_dict['verification_request_perf'] = round(t2[0] - t1[0], 2)
            #performance_dict['verification_request_proc'] = round(t2[1] - t1[1], 2)

            #print(verification_request)

            enc_json.pop('sk')  # Remove secret key from data sent to RE

            print("Send Rating to Reputation Engine")
            t1 = time.perf_counter(), time.process_time()
            rating_request = send_ratinginfo(args.voterSystemID, args.voteeSystemID, padded_rating, padded_count,
                                             voter_reputation_request["R_recent"],
                                             voter_pseudonym_recent["recentpseudonym"],
                                             db_ratingid["voteid"], enc_json, verification_request['signature_rating'],
                                             verification_request['signature_count'], eval_run, iteration).json()
            t2 = time.perf_counter(), time.process_time()
            performance_dict['rating_request_perf'] = round(t2[0] - t1[0], 2)
            #performance_dict['rating_request_proc'] = round(t2[1] - t1[1], 2)

        else:
            #VE1,VR1

            performance_dict['VoteeRep'] = votee_reputation_request["total_count"]
            #performance_dict['VoterRep'] = voter_reputation_request["total_count"]

            voter_enc_keys = voter_reputation_request["enc_keys"]

            key_dir = get_keydirectory_path(str(args.voterSystemID))
            HE_voter = Pyfhel()
            HE_voter.load_context(key_dir + "/context")  # Context and Secret Key stored at Voter
            HE_voter.load_secret_key(key_dir + "/sec.key")
            HE_voter.from_bytes_public_key(voter_enc_keys['pk'].encode('cp437'))
            HE_voter.from_bytes_relin_key(voter_enc_keys['rlk'].encode('cp437'))
            HE_voter.from_bytes_rotate_key(voter_enc_keys['rtk'].encode('cp437'))

            t1 = time.perf_counter(), time.process_time()
            avg_aggr_rep_lvl = compute_rep_level(HE_voter, voter_reputation_request["R_recent"], voter_reputation_request["plain_r_recent"])
            t2 = time.perf_counter(), time.process_time()
            performance_dict['avg_aggr_rep_lvl_perf'] = round(t2[0] - t1[0], 2)

            eq_class1 = config["Rating"]["eq_classes"][0]
            eq_class2 = config["Rating"]["eq_classes"][1]
            eq_class3 = config["Rating"]["eq_classes"][2]
            if eq_class1[0] < avg_aggr_rep_lvl <= \
                    eq_class1[
                        1]:
                eq_class = 0
            elif eq_class2[0] < avg_aggr_rep_lvl <= \
                    eq_class2[
                        1]:
                eq_class = 1
            elif eq_class3[0] < avg_aggr_rep_lvl <= \
                    eq_class3[1]:
                eq_class = 2
            else:
                return print('ERROR: Can not assign correct eq_class')

            t1 = time.perf_counter(), time.process_time()
            padded_rating, padded_count = compute_padded_rating(args.rating, args.ratingNumbers, args.ratingFeatures,
                                                                eq_class)
            t2 = time.perf_counter(), time.process_time()
            performance_dict['padded_rating_perf'] = round(t2[0] - t1[0], 2)

            #print(padded_rating, padded_count)

            plain_rating = np.array(padded_rating)
            plain_count = np.array(padded_count)

            t1 = time.perf_counter(), time.process_time()
            enc_rating = HE_votee.encrypt(plain_rating)
            t2 = time.perf_counter(), time.process_time()
            performance_dict['enc_rating_perf'] = round(t2[0] - t1[0], 2)
            t1 = time.perf_counter(), time.process_time()
            enc_count = HE_votee.encrypt(plain_count)
            t2 = time.perf_counter(), time.process_time()
            performance_dict['enc_count_perf'] = round(t2[0] - t1[0], 2)

            tracemalloc.start()
            start_memory = tracemalloc.get_traced_memory()[0]
            votee_s_context = HE_votee.to_bytes_context()
            end_memory = tracemalloc.get_traced_memory()[0]
            tracemalloc.stop()
            memory_usage_diff = end_memory - start_memory
            performance_dict['size_votee_s_context'] = memory_usage_diff

            tracemalloc.start()
            start_memory = tracemalloc.get_traced_memory()[0]
            votee_s_public_key = HE_votee.to_bytes_public_key()
            end_memory = tracemalloc.get_traced_memory()[0]
            tracemalloc.stop()
            memory_usage_diff = end_memory - start_memory
            performance_dict['size_votee_s_public_key'] = memory_usage_diff

            tracemalloc.start()
            start_memory = tracemalloc.get_traced_memory()[0]
            votee_s_secret_key = HE_votee.to_bytes_secret_key()  # FOR CHECKING IF ENC AND PLAIN CORRECT
            end_memory = tracemalloc.get_traced_memory()[0]
            tracemalloc.stop()
            memory_usage_diff = end_memory - start_memory
            performance_dict['size_votee_s_secret_key'] = memory_usage_diff

            tracemalloc.start()
            start_memory = tracemalloc.get_traced_memory()[0]
            s_enc_rating = enc_rating.to_bytes()
            end_memory = tracemalloc.get_traced_memory()[0]
            tracemalloc.stop()
            memory_usage_diff = end_memory - start_memory
            performance_dict['size_s_enc_rating'] = memory_usage_diff

            tracemalloc.start()
            start_memory = tracemalloc.get_traced_memory()[0]
            s_enc_count = enc_count.to_bytes()
            end_memory = tracemalloc.get_traced_memory()[0]
            tracemalloc.stop()
            memory_usage_diff = end_memory - start_memory
            performance_dict['size_s_enc_count'] = memory_usage_diff

            tracemalloc.start()
            start_memory = tracemalloc.get_traced_memory()[0]
            voter_s_context = HE_voter.to_bytes_context()
            end_memory = tracemalloc.get_traced_memory()[0]
            tracemalloc.stop()
            memory_usage_diff = end_memory - start_memory
            performance_dict['size_voter_s_context'] = memory_usage_diff

            tracemalloc.start()
            start_memory = tracemalloc.get_traced_memory()[0]
            voter_s_public_key = HE_voter.to_bytes_public_key()
            end_memory = tracemalloc.get_traced_memory()[0]
            tracemalloc.stop()
            memory_usage_diff = end_memory - start_memory
            performance_dict['size_voter_s_public_key'] = memory_usage_diff

            tracemalloc.start()
            start_memory = tracemalloc.get_traced_memory()[0]
            voter_s_secret_key = HE_voter.to_bytes_secret_key()  # FOR CHECKING IF ENC AND PLAIN CORRECT
            end_memory = tracemalloc.get_traced_memory()[0]
            tracemalloc.stop()
            memory_usage_diff = end_memory - start_memory
            performance_dict['size_voter_s_secret_key'] = memory_usage_diff

            tracemalloc.start()
            start_memory = tracemalloc.get_traced_memory()[0]
            voter_s_relin_key = HE_voter.to_bytes_relin_key()
            end_memory = tracemalloc.get_traced_memory()[0]
            tracemalloc.stop()
            memory_usage_diff = end_memory - start_memory
            performance_dict['size_voter_s_relin_key'] = memory_usage_diff

            tracemalloc.start()
            start_memory = tracemalloc.get_traced_memory()[0]
            voter_s_rotate_key = HE_voter.to_bytes_rotate_key()
            end_memory = tracemalloc.get_traced_memory()[0]
            tracemalloc.stop()
            memory_usage_diff = end_memory - start_memory
            performance_dict['size_voter_s_rotate_key'] = memory_usage_diff

            votee_enc_json = {
                'context': votee_s_context.decode('cp437'),
                'sk': votee_s_secret_key.decode('cp437'),
                'pk': votee_s_public_key.decode('cp437')
            }

            voter_enc_json = {
                'context': voter_s_context.decode('cp437'),
                'pk': voter_s_public_key.decode('cp437'),
                'sk': voter_s_secret_key.decode('cp437'),
                'rlk': voter_s_relin_key.decode('cp437'),
                'rtk': voter_s_rotate_key.decode('cp437'),
                's_enc_rating': s_enc_rating.decode('cp437'),
                's_enc_count': s_enc_count.decode('cp437'),
            }

            t1 = time.perf_counter(), time.process_time()
            print("Send Rating and signed encrypted voter level to Verification Engine")
            verification_request = send_for_verification(verification_pseudonym_recent["recentpseudonym"],
                                                         voter_reputation_request["R_recent"],
                                                         voter_reputation_request["r_sum_signs"],
                                                         voter_reputation_request["f_sum_signs"], votee_enc_json,
                                                         voter_enc_json, eval_run, iteration).json()
            t2 = time.perf_counter(), time.process_time()
            performance_dict['verification_request_perf'] = round(t2[0] - t1[0], 2)
            #print(verification_request)

            voter_enc_json.pop('sk')  # Remove secret key from data sent to RE

            t1 = time.perf_counter(), time.process_time()
            print("Send Rating to Reputation Engine")
            # print(args.SystemID, args.voteeSystemID, args.rating, reputation_request["R_recent"], pseudonym_request["recentpseudonym"], db_ratingid["voteid"])
            # response = send_ratinginfo(args.SystemID, args.voteeSystemID, padded_rating, padded_count, reputation_request["R_recent"], pseudonym_request["recentpseudonym"], db_ratingid["voteid"])
            rating_request = send_ratinginfo(args.voterSystemID, args.voteeSystemID, padded_rating, padded_count,
                                             voter_reputation_request["R_recent"],
                                             voter_pseudonym_recent["recentpseudonym"],
                                             db_ratingid["voteid"], voter_enc_json,
                                             verification_request['signature_rating'],
                                             verification_request['signature_count'], eval_run, iteration).json()
            t2 = time.perf_counter(), time.process_time()
            performance_dict['rating_request_perf'] = round(t2[0] - t1[0], 2)

    #check_flag = check_equivalence(args.voteeSystemID, rating_request["Updated_Rep"])
    #print(rating_request, f"Ratings equivalent? {check_flag}")
    t100 = time.perf_counter(), time.process_time()
    performance_dict['rating_process_perf'] = round(t100[0] - t0[0], 2)
    performance_dict['rating_process_proc'] = round(t100[1] - t0[1], 2)

    return args.voteeSystemID, rating_request["Updated_Rep"], performance_dict

def compute_rep_level(HE_client, r_recent, plain_r_recent):
    r_sum = [PyCtxt(pyfhel=HE_client, bytestring=x.encode('cp437')) for x in r_recent['r_sum']]
    f_sum = [PyCtxt(pyfhel=HE_client, bytestring=x.encode('cp437')) for x in r_recent['f_sum']]

    r_sum = [np.round(HE_client.decryptFrac(x)[:1],2) for x in r_sum]
    f_sum = [np.round(HE_client.decryptFrac(y)[:1],2) for y in f_sum]

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
        if np.round(r_sum[i],2) != round(plain_r_sum[i],2):
            print(f"Values at index {i} are different: {r_sum[i]} and {plain_r_sum[i]}")
        
        """if r_sum[i] == plain_r_sum[i]:
            print(f"Values at index {i} are the same: {r_sum[i]}")
        else:
            print(f"Values at index {i} are different: {r_sum[i]} and {plain_r_sum[i]}")"""

    for i in range(min(len(f_sum), len(plain_f_sum))):
        if np.round(f_sum[i],2) != round(plain_f_sum[i],2):
            print(f"Values at index {i} are different: {f_sum[i]} and {plain_f_sum[i]}")
        
        """if f_sum[i] == plain_f_sum[i]:
            print(f"Values at index {i} are the same: {f_sum[i]}")
        else:
            print(f"Values at index {i} are different: {f_sum[i]} and {plain_f_sum[i]}")"""

    if avg_aggr_reputation != plain_avg_aggr_reputation:
        print(f"WARNING: Average reputation DIFFERENT")

    return avg_aggr_reputation

def compute_padded_rating(rating, num, features, eq_class):
    sub_features = features[:num[0]] #extract subjective features from all voter features
    rating_dict = {sub_features[i]: rating[i] for i in range(len(sub_features))} #make dict out of voter features and associated rating
    sys_sub_features = [config["Rating"]["rating_fields"][y] for y in range(config["Rating"]["sub_num"])] #extract subjective features from system-wide features
    sub_list = [] #list with subjective voter ratings and paddings for values in other eq_classes
    count_list = [] #list containing number of votes
    for eq in range(config["Rating"]["eq_classes_num"]):
        for feature in sys_sub_features:
            if eq_class == eq and feature in rating_dict.keys():
                sub_list.append(rating_dict[feature]) #put voter rating in correct field in padded rating
                count_list.append(1) #increment only for fields present in voter rating
            else:
                sub_list.append(0)
                count_list.append(0)

    return sub_list, count_list

def get_keydirectory_path(user):
    # Get the current directory where app.py is located
    current_directory = os.path.dirname(os.path.abspath(__file__))

    # Create the path for the specified directory within the 'keys' directory
    directory_path = os.path.join(current_directory, 'votee', 'keys', user)

    return directory_path

def check_equivalence(votee, upd_rep):
    client = MongoClient(config["ReputationManager"]["db_host"], config["ReputationManager"]["db_port"])

    db = client[config["ReputationManager"]["db"]]
    collection = db[config["ReputationManager"]["collection"]]

    key_dir = get_keydirectory_path(str(votee))
    HE_server = Pyfhel()  # Empty creation
    HE_server.load_context(key_dir + "/context")
    HE_server.load_public_key(key_dir + "/pub.key")
    HE_server.load_secret_key(key_dir + "/sec.key")

    rating = collection.find_one({'SystemID': str(votee)})

    rx = PyCtxt(pyfhel=HE_server, bytestring=rating[upd_rep]["enc_sub"]['enc_rating'].encode('cp437'))
    cx = PyCtxt(pyfhel=HE_server, bytestring=rating[upd_rep]["enc_sub"]['enc_count'].encode('cp437'))

    enc_rating = np.round(HE_server.decryptFrac(rx)[0:(3*config['Rating']['sub_num'])], 2)
    enc_count = np.round(HE_server.decryptFrac(cx)[0:(3*config['Rating']['sub_num'])], 2)

    for i in range(min(len(enc_rating), len(rating[upd_rep]["sub"]["rating"]))):
        if np.round(enc_rating[i],2) != round(rating[upd_rep]["sub"]["rating"][i],2):
            print(f"Rating values at index {i} are different: enc_rating {enc_rating[i]} and plain_rating {rating[upd_rep]['sub']['rating'][i]}")
            return False, enc_rating, enc_count, rating[upd_rep]['sub']['rating'], rating[upd_rep]['sub']['count']
        
        """if enc_rating[i] == rating[upd_rep]["sub"]["rating"][i]:
            print(f"Rating values at index {i} are the same: enc_rating {enc_rating[i]} and plain_rating {rating[upd_rep]['sub']['rating'][i]}")
        else:
            print(f"Rating values at index {i} are different: enc_rating {enc_rating[i]} and plain_rating {rating[upd_rep]['sub']['rating'][i]}")
            return False"""

    for i in range(min(len(enc_count), len(rating[upd_rep]["sub"]["count"]))):
        if np.round(enc_count[i],2) != round(rating[upd_rep]["sub"]["count"][i],2):
            print(f"Count values at index {i} are different: enc_count {enc_count[i]} and plain_count {rating[upd_rep]['sub']['count'][i]}")
            return False, enc_rating, enc_count, rating[upd_rep]['sub']['rating'], rating[upd_rep]['sub']['count']
        
        """if enc_count[i] == rating[upd_rep]["sub"]["count"][i]:
            print(f"Count values at index {i} are the same: enc_count {enc_count[i]} and plain_count {rating[upd_rep]['sub']['count'][i]}")
        else:
            print(f"Count values at index {i} are different: enc_count {enc_count[i]} and plain_count {rating[upd_rep]['sub']['count'][i]}")
            return False"""

    return True, enc_rating, enc_count, rating[upd_rep]['sub']['rating'], rating[upd_rep]['sub']['count']

if __name__ == "__main__":
    # Define the evaluation name
    evaluation_name = 'eval1'

    # Define the path to the "eval_data" folder
    eval_data_folder = os.path.join(os.getcwd(), 'eval_data')

    # Create a folder for the evaluation if it doesn't exist within "eval_data"
    evaluation_folder = os.path.join(eval_data_folder, evaluation_name)
    os.makedirs(evaluation_folder, exist_ok=True)

    # Create an empty DataFrame outside the loop
    result_df = pd.DataFrame(columns=['Iteration'])  # Initialize with 'Iteration' column

    for iteration in range(1000):
        # Check if the file exists within the evaluation folder
        file_path = os.path.join(evaluation_folder, 'rating_process.csv')
        file_exists = os.path.isfile(file_path)

        votee, rating, performance_dict = main(iteration, evaluation_name)
        #mem_usage, (votee, rating, performance_dict) = memory_usage((main, (iteration, evaluation_name)), retval=True)

        check_flag, enc_rating, enc_count, plain_rating, plain_count = check_equivalence(votee, rating)
        performance_dict['check'] = check_flag
        performance_dict['enc_rating'] = [enc_rating]
        performance_dict['plain_rating'] = [plain_rating]
        performance_dict['enc_count'] = [enc_count]
        performance_dict['plain_count'] = [plain_count]

        known_columns = ['eval',
                        'iteration',
                        'voter',
                        'votee',
                        'sub_num',
                        'obj_num',
                        'tru_num',
                        'rating_features',
                        'rating_types',
                        'rating_limits',
                        'voter_pseudonym_recent_perf',
                        'votee_pseudonym_recent_perf',
                        'verification_pseudonym_recent_perf',
                        'voter_reputation_request_perf',
                        'votee_reputation_request_perf',
                        'HE_votee.keyGen()_request_perf',
                        'HE_votee.relinKeyGen()_perf',
                        'HE_votee.rotateKeyGen()_perf',
                        'VoteeRep',
                        'VoterRep',
                        'padded_rating_perf',
                        'enc_rating_perf',
                        'enc_count_perf',
                        'size_votee_s_context',
                        'size_votee_s_public_key',
                        'size_votee_s_secret_key',
                        'size_votee_s_relin_key',
                        'size_votee_s_rotate_key',
                        'size_s_enc_rating',
                        'size_s_enc_count',
                        'size_voter_s_context',
                        'size_voter_s_public_key',
                        'size_voter_s_secret_key',
                        'size_voter_s_relin_key',
                        'size_voter_s_rotate_key',
                        'enc_keys_perf',
                        'inq_enc_keys_perf',
                        'km_pseudonym_recent_perf',
                        'enc_sk_request_perf',
                        'fernet_key_perf',
                        'fernet_cipher_perf',
                        'decrypted_sk_perf',
                        'sk_send_perf',
                        'rating_process_perf',
                        'rating_process_proc',
                        'check',
                        'enc_rating',
                        'plain_rating',
                        'enc_count',
                        'plain_count',
                        'sub_rating']
        df_iteration = pd.DataFrame(columns=known_columns)
        data_dict = {col: [performance_dict.get(col, None)] for col in known_columns}
        df_to_append = pd.DataFrame(data_dict)
        df_iteration = pd.concat([df_iteration, df_to_append], ignore_index=True, sort=False)

        # Save the df_iteration DataFrame to the CSV file (append mode if the file exists)
        df_iteration.to_csv(file_path, mode='a', header=not file_exists, index=False)

        if check_flag is False:
            break