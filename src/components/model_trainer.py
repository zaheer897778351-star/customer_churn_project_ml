import os
from src.utils.exception import CustomException
from src.utils.logger import logging
import sys
from src.utils.helper import save_object
from src.utils.helper import evaluate_model
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    model_saved_file_path = os.path.join('artifacts',"model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def ModelTrainer_method(self,x_train,x_test,Y_train,Y_test):
        try:
            logging.info("now entered model trainer")
            X_train = x_train
            X_test = x_test
            y_train = Y_train
            y_test = Y_test

            models = {
            "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    
            "DecisionTree": DecisionTreeClassifier(class_weight="balanced"),
    
            "RandomForest": RandomForestClassifier(class_weight="balanced", n_estimators=200),
    
            "GradientBoosting": GradientBoostingClassifier(),
    
            "AdaBoost": AdaBoostClassifier(),
    
            "SVM (RBF)": SVC(kernel='rbf', class_weight="balanced"),
    
            "KNN": KNeighborsClassifier()
            }

            model_reports:dict = evaluate_model(X_train=X_train,X_test=X_test,y_train=y_train,y_test=y_test,models=models)
            best_model_socre = max(sorted(model_reports.values()))
            best_model_name = list(models.keys())[
                list(model_reports.values()).index(best_model_socre)
            ]
            best_model = models[best_model_name]
            if best_model_socre < 0.6:
                raise CustomException("no best model found ")
            logging.info("best model found")

            predicted = best_model.predict(X_test)
            r_score = accuracy_score(y_test,predicted)
            logging.info("model training completed")

            save_object(file_path=self.model_trainer_config.model_saved_file_path,
                        obj=best_model)

            return(
                r_score,best_model
            )


        except Exception as e:
            raise CustomException(e,sys)
