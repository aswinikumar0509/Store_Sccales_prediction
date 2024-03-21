from Model_Building.logger import logging
import os,sys
from Model_Building.exception import MartException
from Model_Building.entity import config_entity
from Model_Building.entity import artifact_entity
from Model_Building import utils
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from imblearn.combine import SMOTETomek
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
ll= LabelEncoder()

class DataTransformation:

    def __init__(self,data_transformation_config:config_entity.DataTransformationConfig,data_ingestion_artifact:artifact_entity.DataIngestionArtifact):

        try:
            logging.info(f"{'>>'*20} Data Transformation {'<<'*20}")
            self.data_transformation_config = data_transformation_config
            self.data_ingestion_artifact = data_ingestion_artifact
        except Exception as e:
            raise MartException(e,sys)
        
    @classmethod
    def get_data_transformation_object(cls)->Pipeline:
        try:
            Simple_Imputer = SimpleImputer(strategy = 'constant',fill_value = 0)
            Robust_Scaler = RobustScaler()
            pipeline = Pipeline(steps = [('Imputer',Simple_Imputer),('RobustScaler',Robust_Scaler)])
            return pipeline
        except Exception as e:
            raise MartException(e,sys)


    def initate_data_transformation(self,)->artifact_entity.DataTransformationArtifact:

        try:

            # Reading training and testing file
            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
            # Handiling the missing value
            train_df["Item_Outlet_Sales"].fillna(train_df['Item_Outlet_Sales'].mean() , inplace = True)
            train_df["Item_Weight"].fillna(train_df["Item_Weight"].mean(), inplace=True)
            train_df["Outlet_Size"].fillna(train_df["Outlet_Size"].mode()[0], inplace=True)
            train_df.drop(["Item_Identifier", "Outlet_Identifier"], axis=1, inplace=True)
            # label encoding the feature columns

            lable_encoder = LabelEncoder()
            train_df['Item_Fat_Content'] = lable_encoder.fit_transform(train_df['Item_Fat_Content'])
            train_df['Item_Type'] = lable_encoder.fit_transform(train_df['Item_Type'])
            train_df['Outlet_Size'] = lable_encoder.fit_transform(train_df['Outlet_Size'])
            train_df['Outlet_Location_Type'] = lable_encoder.fit_transform(train_df['Outlet_Location_Type'])
            train_df['Outlet_Type'] = lable_encoder.fit_transform(train_df['Outlet_Type'])
                        

            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            # Handaling teh missing value
            test_df["Item_Weight"].fillna(train_df["Item_Weight"].mean(), inplace=True)
            test_df["Outlet_Size"].fillna(train_df["Outlet_Size"].mode()[0], inplace=True)
            test_df.drop(["Item_Identifier", "Outlet_Identifier"], axis=1, inplace=True)

            # Performing label encoding

            test_df['Item_Fat_Content'] = lable_encoder.fit_transform(test_df['Item_Fat_Content'])
            test_df['Item_Type'] = lable_encoder.fit_transform(test_df['Item_Type'])
            #test_df['Outlet_Size'] = lable_encoder.fit_transform(test_df['Outlet_Size'])
            test_df['Outlet_Location_Type'] = lable_encoder.fit_transform(test_df['Outlet_Location_Type'])
            test_df['Outlet_Type'] = lable_encoder.fit_transform(test_df['Outlet_Type'])

            # Selecting the input feature for test and train
            input_feature_train_df = train_df.drop(['Item_Outlet_Sales'],axis = 1)
            input_feature_test_df = test_df

            # target feature for test and train dataframe

            target_feature_train_arr = train_df['Item_Outlet_Sales']

            target_feature_test_arr = test_df['Item_Outlet_Sales']

            transformation_pipeline = DataTransformation.get_data_transformation_object()
            transformation_pipeline.fit(input_feature_train_df)

            # Transformation
                        
            input_feature_train_arr = transformation_pipeline.transform(input_feature_train_df)
           # input_feature_test_arr = transformation_pipeline.transform(input_feature_test_df)

            smt = SMOTETomek(random_state=42)
            logging.info(f"Before resampling in training set Input: {input_feature_train_arr.shape} Target:{target_feature_train_arr.shape}")
            #input_feature_train_arr, target_feature_train_arr = smt.fit_resample(input_feature_train_arr, target_feature_train_arr)
            #logging.info(f"After resampling in training set Input: {input_feature_train_arr.shape} Target:{target_feature_train_arr.shape}")
           
            #target encoder
            train_arr = np.c_[input_feature_train_arr, target_feature_train_arr ]
            #test_arr = np.c_[input_feature_test_arr,target_feature_train_arr]


            #save numpy array
            utils.save_numpy_array_data(file_path=self.data_transformation_config.transformed_train_path,
                                        array=train_arr)


            utils.save_object(file_path=self.data_transformation_config.transform_object_path,obj=transformation_pipeline)

           # utils.save_object(file_path=self.data_transformation_config.target_encoder_path,obj=lable_encoder)



            data_transformation_artifact = artifact_entity.DataTransformationArtifact(
                transform_object_path=self.data_transformation_config.transform_object_path,
                transformed_train_path = self.data_transformation_config.transformed_train_path,
                transformed_test_path = self.data_transformation_config.transformed_test_path,
                #target_encoder_path = self.data_transformation_config.target_encoder_path

            )

            logging.info(f"Data transformation object {data_transformation_artifact}")
            return data_transformation_artifact
        except Exception as e:
            raise MartException(e,sys)