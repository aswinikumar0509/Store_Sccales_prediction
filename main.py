from Model_Building.logger import logging
from Model_Building.exception import MartException
import os,sys
from Model_Building.entity import config_entity
from Model_Building.entity import artifact_entity
import pandas as pd
import numpy as np
from Model_Building import utils
from Model_Building.component.data_ingestion import DataIngestion
from Model_Building.component.data_validation import DataValidation
from Model_Building.component.data_transformation import DataTransformation
from Model_Building.component.model_trainer import ModelTrainer
from Model_Building.component.model_evaluation import ModelEvalutaion
from Model_Building.component.model_pusher import ModelPusher


print(__name__)
if __name__=="__main__":
     try:
          training_pipeline_config = config_entity.TrainingPipelineConfig()

          #data ingestion
          data_ingestion_config  = config_entity.DataIngestionConfig(training_pipeline_config=training_pipeline_config)
          print(data_ingestion_config.to_dict())
          data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
          data_ingestion_artifact = data_ingestion.initate_data_ingestion()

          # data validation
          data_validation_config = config_entity.DataValidationConfig(training_pipeline_config=training_pipeline_config)
          data_validation = DataValidation(data_validation_config=data_validation_config,
                        data_ingestion_artifact=data_ingestion_artifact)

          data_validation_artifact = data_validation.initiate_data_validation()

          # data transformation
          data_transformation_config = config_entity.DataTransformationConfig(training_pipeline_config=training_pipeline_config)
          data_transformation = DataTransformation(data_transformation_config=data_transformation_config, 
          data_ingestion_artifact=data_ingestion_artifact)
          data_transformation_artifact = data_transformation.initate_data_transformation()

          #model trainer
          model_trainer_config = config_entity.ModelTrainerConfig(training_pipeline_config=training_pipeline_config)
          model_trainer = ModelTrainer(model_trainer_config=model_trainer_config, data_transformation_artifact=data_transformation_artifact)
          model_trainer_artifact = model_trainer.initiate_model_trainer()

          #model evaluation
          model_eval_config = config_entity.ModelEvaluationConfig(training_pipeline_config=training_pipeline_config)
          model_eval  = ModelEvalutaion(model_evaluation_config=model_eval_config,
                                        data_ingestion_artifact=data_ingestion_artifact,
                                        data_transformation_artifact=data_transformation_artifact,
                                        model_trainer_artifact=model_trainer_artifact)
          model_eval_artifact = model_eval.initiate_model_evaluation()

          # model Pusher
          model_pusher_config = config_entity.ModelPusherConfig(training_pipeline_config)
        
          model_pusher = ModelPusher(model_pusher_config=model_pusher_config, 
                data_transformation_artifact=data_transformation_artifact,
                model_trainer_artifact=model_trainer_artifact)

          model_pusher_artifact = model_pusher.initiate_model_pusher()


     except Exception as e:
          print(e) 