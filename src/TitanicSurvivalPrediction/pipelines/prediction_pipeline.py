import os
import sys
import pandas as pd
import pickle
from src.TitanicSurvivalPrediction.exeception import customException
from src.TitanicSurvivalPrediction.logger import logging

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,input_data):
        try:
            model_path=os.path.join('artifacts','model.pkl')

            model=pickle.load(model_path)

            pred=model.predict(input_data)
            return pred
        
        except Exception as e:
            raise customException(e,sys)

class CustomData:
    def __init__(self,
                 Pclass:int,
                 Name:str,
                 Sex:str,
                 Age:float,
                 SibSp:int,
                 Ticket:object,
                 Fare:float,
                 Cabin:object,
                 Embarked:str
                 ):
        
        self.Pclass=Pclass
        self.Name=Name
        self.Sex=Sex
        self.Age=Age
        self.SibSp=SibSp
        self.Ticket=Ticket
        self.Fare=Fare
        self.Cabin=Cabin
        self.Embarked=Embarked
    
    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict={
                'Pclass':[self.Pclass],
                'Name':[self.Name],
                'Sex':[self.Sex],
                'Age':[self.Age],
                'SibSp':[self.SibSp],
                'Ticket':[self.Ticket],
                'Fare':[self.Fare],
                'Cabin':[self.Cabin],
                'Embarked':[self.Embarked]
            }
            data_frame=pd.DataFrame(custom_data_input_dict)
            logging.info(f"DataFrame Gathered")
            return data_frame
        except Exception as e:
            logging.info(f"Exception Occured in prediction pipeline")
            raise customException(e,sys)