import os
from src.utils.exception import CustomException
from src.utils.logger import logging
import pandas as pd
import sys
from src.utils.config_loader import load_config
from src.components.data_transformation import DataTransformation

from src.components.model_trainer import ModelTrainer

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path:str = os.path.join('artifacts',"train.csv")
    test_data_path:str = os.path.join('artifacts',"test.csv")
    raw_data_path:str = os.path.join('artifacts',"raw.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        self.config = load_config()

    def data_ingestion(self):
        
        try:
            logging.info("Now Entered into data ingestion part")
            df = pd.read_csv('notebook/customer_churn_updated.csv')
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            logging.info("now created artifacts folder and raw csv file")

            # print(type(self.config))
            # print(self.config)

            test_size = self.config["model_trainer"]["test_size"]
            random_state = self.config["model_trainer"]["random_state"]

            train_set,test_set = train_test_split(df,test_size=test_size,random_state=random_state)
            logging.info("now entered into train ,test set data ")

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            
            logging.info("now train ,test set data created")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )



        except Exception as e:
            raise CustomException(e,sys)
        
if __name__ =="__main__":
    obj=DataIngestion()
    train , test = obj.data_ingestion()
    data_trans = DataTransformation()
    X_train,X_test,y_train,y_test = data_trans.initiate_data_transform(train,test)
    model = ModelTrainer()
    print(model.ModelTrainer_method(X_train,X_test,y_train,y_test))