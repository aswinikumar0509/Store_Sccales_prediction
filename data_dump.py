import pymongo
import pandas as pd
import json
import os , sys
from Model_Building.config import mongo_client

Data_FILE_PATH="Train.csv"
TEST_FILE_PATH="Test.csv"
DATABASE_NAME="StoreSales"
COLLECTION_NAME="Sales"

if __name__=="__main__":
    # For the train dataset
    train_df = pd.read_csv(Data_FILE_PATH)
    print(f"Rows and columns: {train_df.shape}")

    #Convert dataframe to json so that we can dump these record in mongo db
    train_df.reset_index(drop=True,inplace=True)

    json_record = list(json.loads(train_df.T.to_json()).values())
    print(json_record[0])

    #Convert dataframe to json so that we can dump these record in mongo db
    train_df.reset_index(drop=True,inplace=True)

    #insert converted json record to mongo db
    mongo_client[DATABASE_NAME][COLLECTION_NAME].insert_many(json_record)

    test_df = pd.read_csv(TEST_FILE_PATH)
    print(f"Rows and columns: {test_df.shape}")

    #Convert dataframe to json so that we can dump these record in mongo db
    test_df.reset_index(drop=True,inplace=True)

    json_record = list(json.loads(test_df.T.to_json()).values())
    print(json_record[0])

    #Convert dataframe to json so that we can dump these record in mongo db
    test_df.reset_index(drop=True,inplace=True)

    #insert converted json record to mongo db
    mongo_client[DATABASE_NAME][COLLECTION_NAME].insert_many(json_record)