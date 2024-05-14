import pandas as pd
import numpy as np
import os
import sys
from src.TitanicSurvivalPrediction.logger import logging
from src.TitanicSurvivalPrediction.exeception import customException

from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from pathlib import Path

class DataIngestionConfig:
    data_path:str=os.path.join('artifacts')
    train_data_path:str=os.path.join('artifacts','train_data.csv')
    test_data_path:str=os.path.join('artifacts','test_data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
    
    def initiate_data_ingection(self):
        logging.info("Data Ingestion Started")

        try:
            train_data=pd.read_csv(r"C:\Users\DHINESH BABBU A\Documents\Full Stack Data Science\ML_for_titanic\notebooks\data\train.csv")
            logging.info('I have Read the train data as the DataFrame')

            os.makedirs(os.path.join(self.ingestion_config.data_path),exist_ok=True)
            train_data.to_csv(self.ingestion_config.train_data_path,index=False)
            logging.info("I have Saved the train csv data in artifacts folder sucessfully")

            test_csv=pd.read_csv(r"C:\Users\DHINESH BABBU A\Documents\Full Stack Data Science\ML_for_titanic\notebooks\data\test.csv")
            logging.info('I have Read the test data as the DataFrame')

            test_csv.to_csv(self.ingestion_config.test_data_path,index=False)
            logging.info("I have Saved the test csv data in artifacts folder sucessfully")

            logging.info('Data Ingestion Part Completed Sucessfully')

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        
        except Exception as e:
            logging.info("Exception during occured at data ingestion stage!!!")
            raise customException(e,sys)


