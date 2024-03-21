from Model_Building.logger import logging
import os,sys
from Model_Building.exception import MartException
from Model_Building.entity import config_entity
from Model_Building.entity import artifact_entity
from Model_Building import utils
import pandas as pd
import numpy as np
import os,sys 
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


class ModelTrainer:

    def __init__(self,model_trainer_config:config_entity.ModelTrainerConfig,data_transformation_artifact:artifact_entity.DataTransformationArtifact):

        try:
            logging.info(f"{'>>'*20} Model Trainer {'<<'*20}")
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise MartException(e,sys)

    def fine_tune(self):
        try:
            pass
        except Exception as e:
            raise MartException(e,sys)
        
    def train_model(self,x,y):
        try:
            rf_re = RandomForestRegressor()
            rf_re.fit(x,y)
            return rf_re
        except Exception as e:
            raise MartException(e,sys)

    def initiate_model_trainer(self,)->artifact_entity.ModelTrainerArtifact:

        try:
            logging.info(f"loading trian and test array")

            train_arr = utils.load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_train_path)
            #test_arr = utils.load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_test_path)

           
            logging.info(f"Splitting the data into train and test array")
           # x = train_arr[:,:-1]
           # y = train_arr[:,-1]
           #X_train,y_train,X_test,y_test = train_test_split(x,y,test_size=0.2)
            x_train,y_train = train_arr[:,:-1],train_arr[:,-1]
            x_test,y_test = train_arr[:,:-1],train_arr[:,-1]




            logging.info(f" Train the Model")
            model = self.train_model(x = x_train, y = y_train)

            logging.info(f"Calculating the train accuracy")
            yhat_train = model.predict(x_train)
            r2_train_score = r2_score(y_true=y_train,y_pred=yhat_train)
            rms = sqrt(mean_squared_error(y_true=y_train, y_pred=yhat_train))

            logging.info(f"calculating the test accuracy")
            yhat_test = model.predict(x_test)
            r2_test_score = r2_score(y_true=y_test,y_pred=yhat_test)
            rms = sqrt(mean_squared_error(y_true=y_test, y_pred=yhat_test))

            logging.info(f"test score:{r2_test_score}")

            logging.info("Checking the model is overfitting or not")

            diff = abs(r2_train_score-r2_test_score)

            if diff > self.model_trainer_config.overfitting_threshold:
                raise Exception(f"Train and Test score diff :{diff} is more than the overfitting threshold {self.model_trainer_config.overfitting_threshold}")

            # save the model in the artifact
            logging.info(f"Saving model object ")
            utils.save_object(file_path=self.model_trainer_config.model_path , obj= model)

            # prepare the artifact
            logging.info(f"prepare the artifact")
            model_trainer_artifact = artifact_entity.ModelTrainerArtifact(model_path=self.model_trainer_config.model_path,r2_train_score=r2_train_score,r2_test_score=r2_test_score)

            logging.info(f"Model trainer Artifact:{model_trainer_artifact}")

            return model_trainer_artifact  
        
        except Exception as e:
            raise MartException(e,sys)

    
        