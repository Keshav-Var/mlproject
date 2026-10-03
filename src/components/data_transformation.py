import numpy as np
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
import sys
import os

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

from src.utils import save_object

@dataclass
class DataTransformerConfig:
    preprocessor_obj_file_path:str = os.path.join('artifacts','preprocessor.pkl')

class DataTransformer:
    def __init__(self):
        self.data_transformation_config = DataTransformerConfig()

    def get_data_transformer_obj(self):
        '''This function is responsible for data transformation'''
        try:
            numerical_features = ['reading_score','writing_score']
            categorical_features = ['gender','race_ethnicity','parental_level_of_education','lunch','test_preparation_course']

            num_pipeline = Pipeline(steps=[
                ('imputer',SimpleImputer(strategy='median')),
                ('scaler',StandardScaler())
            ])

            cat_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('oh_encoder',OneHotEncoder()),
                    ('scaler',StandardScaler(with_mean=False))
                ]
            )
            logging.info(f"Catagorical features are {categorical_features}")
            logging.info(f"Numerical features are {numerical_features}")

            preprocessor = ColumnTransformer(
                [('num_transformer',num_pipeline,numerical_features),
                ('cat_transformer',cat_pipeline,categorical_features)]
            )
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Reading of train and test data is completed")
            logging.info("Obtaining preprocessor obj")

            preprocessor_obj = self.get_data_transformer_obj()
            target_col_name = 'math_score'

            input_feature_train_df = train_df.drop(columns=[target_col_name])
            target_feature_train_df = train_df[target_col_name]

            input_feature_test_df = test_df.drop(columns=[target_col_name])
            target_feature_test_df = test_df[target_col_name]

            logging.info("Applying preprocessing object on training dataframe and test dataframe.")
            
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

            save_object(file_path=self.data_transformation_config.preprocessor_obj_file_path,
                        obj=preprocessor_obj)
            
            logging.info("saved preprocessing object.")

            return (train_arr,test_arr,self.data_transformation_config.preprocessor_obj_file_path)

        except Exception as e:
            raise CustomException(e,sys)