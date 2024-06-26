from src.TitanicSurvivalPrediction.components.data_ingestion import DataIngestion
from src.TitanicSurvivalPrediction.components.data_transformation import DataTransformation
from src.TitanicSurvivalPrediction.components.model_training import ModelTraining

obj_ingestion=DataIngestion()
train_data,test_data=obj_ingestion.initiate_data_ingection()

obj_trandformation=DataTransformation()
train_X,train_y=obj_trandformation.initiate_data_transformation(train_data,test_data)

obj_modeltraining=ModelTraining()
obj_modeltraining.initiate_model_training(train_X,train_y)