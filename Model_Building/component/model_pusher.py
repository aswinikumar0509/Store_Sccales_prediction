from Model_Building.logger import logging
from Model_Building.exception import MartException
from Model_Building.entity import config_entity 
from Model_Building.entity import artifact_entity
from Model_Building import utils
import pandas as pd
import numpy as np
import os,sys 
from Model_Building.predictor import ModelResolver
from Model_Building.utils import load_object,save_object
from Model_Building.entity.artifact_entity import DataTransformationArtifact,ModelTrainerArtifact,ModelPusherArtifact
from Model_Building.entity import config_entity

class ModelPusher:

    def __init__(self,model_pusher_config:config_entity.ModelPusherConfig,
                 data_transformation_artifact:DataTransformationArtifact,
                 model_trainer_artifact:ModelTrainerArtifact):
        try:
            logging.info(f"{'>>'*20} Model Pusher {'<<'*20}")
            self.model_pusher_config = model_pusher_config
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_artifact = model_trainer_artifact
            self.model_resolver = ModelResolver(model_registry=self.model_pusher_config.saved_model_dir)
        except Exception as e:
            raise MartException(e,sys)
        
    
    def initiate_model_pusher(self,)->ModelPusherArtifact:
        try:
            # load object
            logging.info(f" Loding transform model and target encoder")
            transformer = load_object(file_path=self.data_transformation_artifact.transform_object_path)
            model = load_object(file_path = self.model_trainer_artifact.model_path)

            # model pusher dir
            logging.info(f"Saving model into model pusher directory")

            save_object(file_path=self.model_pusher_config.pusher_transformer_path, obj=transformer)
            save_object(file_path=self.model_pusher_config.pusher_model_path, obj=model)

           #saved model dir
            logging.info(f"Saving model in saved model dir")
            transformer_path=self.model_resolver.get_latest_save_transformer_path()
            model_path=self.model_resolver.get_latest_save_model_path()

            save_object(file_path=transformer_path, obj=transformer)
            save_object(file_path=model_path, obj=model)
          
            model_pusher_artifact = ModelPusherArtifact(pusher_model_dir=self.model_pusher_config.pusher_model_dir,
            saved_model_dir=self.model_pusher_config.saved_model_dir)
            logging.info(f"Model pusher artifact: {model_pusher_artifact}")
            return model_pusher_artifact
        
        except Exception as e:
            raise MartException(e,sys)