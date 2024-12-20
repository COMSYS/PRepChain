import tomli
import os
import csv
from pymongo import MongoClient
import shutil

#This file allows for the complete (re-)initialization of databases and cleaning up of old keys

# Load configuration from config.toml
with open("config.toml", mode="rb") as fp:
    config = tomli.load(fp)

def drop_collection_if_exists(db_config):
    client = MongoClient(db_config["db_host"], db_config["db_port"])
    db = client[db_config["db"]]
    collection_name = db_config["collection"]

    # Check if the collection exists
    if collection_name in db.list_collection_names():
        # Drop the collection if it exists
        db[collection_name].drop()
        print(f"Collection '{collection_name}' dropped.")

def initialize_collection(db_config, csv_folder, csv_filename):
    client = MongoClient(db_config["db_host"], db_config["db_port"])
    db = client[db_config["db"]]
    collection_name = db_config["collection"]

    # Check if the database already exists
    if db_config["db"] in client.list_database_names():
        # Drop the collection if it exists
        db[collection_name].drop()
        print(f"Collection '{collection_name}' dropped.")

    # Create the collection
    collection = db[collection_name]
    print(f"Collection '{collection_name}' created.")

    csv_file_path = os.path.join(os.path.dirname(__file__), csv_folder, csv_filename)
    with open(csv_file_path, "r", encoding="utf-8-sig") as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=";")
        data_to_insert = []

        for row in csv_reader:
            # Convert "recent_count" and "total_count" to integers if initializing "reputation_manager"
            if collection_name == "reputation_manager":
                row["recent_count"] = int(row.get("recent_count", 0))
                row["total_count"] = int(row.get("total_count", 0))

            data_to_insert.append(row)

    if data_to_insert:
        collection.insert_many(data_to_insert)

# Drop the "reputation_engine" collection if it exists
drop_collection_if_exists(config["ReputationEngine"])

# Example usage:
# Initialize Pseudonym Manager DB
initialize_collection(config["PseudonymManager"], "pseudonym_manager", "governmententity.csv")

# Initialize Reputation Manager DB
initialize_collection(config["ReputationManager"], "reputation_manager", "init_RM_database.csv")

#Delete all key folders if there exist any
def clear_keys_folder(module_name):
    keys_folder_path = os.path.join(module_name, "keys")

    if os.path.exists(keys_folder_path):
        try:
            shutil.rmtree(keys_folder_path)
            print(f"Deleted the '{keys_folder_path}' directory and its contents.")
        except Exception as e:
            print(f"Failed to delete the '{keys_folder_path}' directory: {str(e)}")


# Clear the "keys" folder for reputation_manager
clear_keys_folder("reputation_manager")

# Clear the "keys" folder for key_manager
clear_keys_folder("key_manager")

# Clear the "keys" folder for votee
clear_keys_folder("votee")
