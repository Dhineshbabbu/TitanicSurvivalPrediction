import pandas as pd
import numpy as np
import os
import sys
from src.TitanicSurvivalPrediction.logger import logging
import optuna
from src.TitanicSurvivalPrediction.exeception import customException
import pickle
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


class ModelTrainingConfig:
    training_model_file=os.path.join('artifacts','model.pkl')

class ModelTraining:
    def __init__(self):
        self.model_training_config=ModelTrainingConfig()

    def initiate_model_training(self,train_X,train_y):
        try:
            logging.info(f"Model Training Started")

            logging.info(f"Splitting the data for training and validation")

            X_train,X_val,y_train,y_val=train_test_split(train_X,train_y,test_size=0.3)

            logging.info(f"Performing Hyperparameter tuning for the Classification Algorithm")

            def objective(trail):
                params={
                    'max_depth':trail.suggest_int('max_depth',3,10),
                    'learning_rate':trail.suggest_loguniform('learning_rate',0.001,0.1),
                    'n_estimators':trail.suggest_int('n_estimators',100,1000),
                    'subsample':trail.suggest_float('subsample',0.5,1),
                    'colsample_bytree':trail.suggest_float('colsample_bytree',0.5,1),
                }
                
                model=XGBClassifier(**params)
                
                model.fit(X_train,y_train)
                
                y_pred=model.predict(X_val)
                
                accuracy=accuracy_score(y_val,y_pred)
                
                return accuracy
            
            study=optuna.create_study(direction='maximize')
            study.optimize(objective,n_trials=50)

            best_params=study.best_trial.params

            tuned_model=XGBClassifier(**best_params)
            tuned_model.fit(X_train,y_train)

            logging.info(f"Training accuracy of the model : {tuned_model.score(X_train,y_train)}")
            logging.info(f"Testing accuracy of the model : {accuracy_score(y_val,tuned_model.predict(X_val))}")

            pickle.dump(tuned_model,open(self.model_training_config.training_model_file,'wb'))

            logging.info(f"Model has been saved as pickle file in artifacts sucessfully")

            logging.info(f"Model Training Completed Sucessfully")

        except Exception as e:
            logging.info("Exception occured at Model Training Stage !!!")
            raise customException(e,sys)
            

