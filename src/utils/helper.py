import os
from src.utils.exception import CustomException
from src.utils.logger import logging
import pandas as pd
import sys
import dill
from sklearn.metrics import accuracy_score

def save_object(file_path,obj):
    try:
        dir = os.path.dirname(file_path)
        os.makedirs(dir,exist_ok=True)
        with open(file_path,'wb') as file_obj:
            dill.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e,sys)
    

def evaluate_model(X_train,X_test,y_train,y_test,models):
    try:
        logging.info("entered into model training")
        score = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]

            model.fit(X_train,y_train)

            y_pred_test = model.predict(X_test)

            rank = accuracy_score(y_test,y_pred_test)
            score[list(models.keys())[i]] = rank

        return score
        


    except Exception as e:
        raise CustomException(e,sys)

def load_object(file_path):
    try:
        with open(file_path,"rb") as file_obj:
            return dill.load(file_obj)
        
    except Exception as e:
        raise CustomException(e,sys)