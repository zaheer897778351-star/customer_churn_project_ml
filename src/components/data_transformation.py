import os
from src.utils.exception import CustomException
from src.utils.logger import logging
import pandas as pd
import sys
from src.utils.config_loader import load_config
from src.utils.helper import save_object
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from dataclasses import dataclass

@dataclass

class DataTransformConfig:
    preprocess_file_path = os.path.join('artifacts',"preprocess.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transform = DataTransformConfig()
        self.config = load_config()

    def get_data_transformation_obj(self):
        try:
            num_col = ['Satisfaction Score',
                        'Churn Score',
                        'Tenure in Months',
                        'Monthly Charge',
                        'Total Long Distance Charges',
                        'Number of Referrals',
                        'CLTV',
                        'Age',
                        'Dependents',
                        'Internet Service']
            
            cat_col = ['Gender']

            num_col_pipeline = Pipeline(
                            steps=[
                        ("simpleimputer",SimpleImputer(strategy="median")),
                        ("standarsclaer",StandardScaler())
                                ]
            )
            cat_col_pipeline = Pipeline(
                            steps=
                                [
                            ("imputer",SimpleImputer(strategy="most_frequent")),
                            ("onehotenoder",OneHotEncoder()),
                            ("standardscaler",StandardScaler(with_mean=False))
                            ]
            )
            
            preprocess = ColumnTransformer(
                        [
                        ("num",num_col_pipeline,num_col),
                        ("cat",cat_col_pipeline,cat_col)
                        ]
            )

            return preprocess


        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transform(self,train_path,test_path):
        try:
            logging.info("now entered into the initiate data transform")
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)

            target = self.config["data_transformation"]["target_column"]
            preprocessor = self.get_data_transformation_obj()

            X_train_data = train_data.drop(columns=[target])
            y_train_data = train_data[target]

            X_test_data= test_data.drop(columns=[target])
            y_test_data = test_data[target]

            X_train_processed = preprocessor.fit_transform(X_train_data)
            X_test_processed = preprocessor.transform(X_test_data)

            logging.info("successfully x_train and test processed")

            save_object(file_path=self.data_transform.preprocess_file_path,
                        obj =preprocessor)

            return(
                X_train_processed,
                X_test_processed,
                y_train_data,
                y_test_data
            )


        except Exception as e:
            raise CustomException(e,sys)