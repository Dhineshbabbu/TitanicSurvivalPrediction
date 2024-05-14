import pandas as pd
import numpy as np
import os
import sys
from src.TitanicSurvivalPrediction.logger import logging
from src.TitanicSurvivalPrediction.exeception import customException

from sklearn.preprocessing import LabelEncoder

class DataTransformationConfig:
    preprocessed_test_data:str=os.path.join('artifacts','preprocessed_test_data.csv')

class DataTransformation:
    def __init__(self):
        self.transform_config=DataTransformationConfig()

    def get_data_transformation(self,train_df,test_df):

        logging.info('Replacing NULL values for train data with mean/mode/median')
        train_df.fillna({'Age':train_df['Age'].median()},inplace=True)
        train_df.fillna({'Cabin':train_df['Cabin'].mode().iloc[0]},inplace=True)
        train_df.fillna({'Embarked':train_df['Embarked'].mode().iloc[0]},inplace=True)

        logging.info('Replacing NULL values in test data with mean/mode/median')
        test_df.fillna({'Age':test_df['Age'].median()},inplace=True)
        test_df.fillna({'Cabin':test_df['Cabin'].mode().iloc[0]},inplace=True)
        test_df.fillna({'Fare':test_df['Fare'].median()},inplace=True)

        logging.info('Performing Label Encoding')

        self.encoder=LabelEncoder()
        def label_encoder(col):
            if col.dtypes=='object':
                col=self.encoder.fit_transform(col)
            return col
        
        train_df=train_df.apply(lambda col:label_encoder(col))
        test_df=test_df.apply(lambda col:label_encoder(col))

        train_df.drop('PassengerId',axis=1,inplace=True)
        test_df.drop('PassengerId',axis=1,inplace=True)

        return (
            train_df,
            test_df
        )


    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info(f"Data Transformation Started")

            logging.info("Reading train data and test data were completed")
            logging.info(f"Train Dataframe Head:\n{train_df.head().to_string()}")
            logging.info(f"Test Dataframe Head:\n{test_df.head().to_string()}")

            train_df,test_df=self.get_data_transformation(train_df,test_df)

            test_df.to_csv(self.transform_config.preprocessed_test_data,index=False)
            logging.info(f"Preprocessed test data has been stored in atrifacts sucessfully")

            train_X,train_y=train_df.drop('Survived',axis=1),train_df['Survived']

            logging.info(f"Data Transformation completed sucessfully")

            return(
                train_X,
                train_y
            )
        
        except Exception as e:
            logging.info("Exception occurs in the Data Transformation stage!!")
            raise customException(e,sys)




