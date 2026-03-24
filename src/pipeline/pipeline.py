import sys
import os
import pandas as pd
from src.utils.exception import CustomException
from src.utils.helper import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path = os.path.join('artifacts',"model.pkl")
            preprocessor_path = os.path.join('artifacts',"preprocess.pkl")
            model = load_object(file_path=model_path)
            processor = load_object(file_path=preprocessor_path)
            data_scaled = processor.transform(features)
            preds = model.predict(data_scaled)
            probability = model.predict_proba(data_scaled)
            return (
                preds,probability
            )

        except Exception as e:
            raise CustomException(e,sys)

class CustomData:
    def __init__(self,
                 Satisfaction_Score: int,
                 Churn_Score: int,
                 Tenure_in_Months: int,
                 Monthly_Charge: float,
                 Total_Long_Distance_Charges: float,
                 Number_of_Referrals: int,
                 CLTV: int,
                 Age: int,
                 Dependents: int,
                 Internet_Service: int,
                 Gender: str
                 ):
        
        self.Satisfaction_Score = Satisfaction_Score
        self.Churn_Score = Churn_Score
        self.Tenure_in_Months = Tenure_in_Months
        self.Monthly_Charge = Monthly_Charge
        self.Total_Long_Distance_Charges = Total_Long_Distance_Charges
        self.Number_of_Referrals = Number_of_Referrals
        self.CLTV = CLTV
        self.Age = Age
        self.Dependents = Dependents
        self.Internet_Service = Internet_Service
        self.Gender = Gender

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Satisfaction Score": [self.Satisfaction_Score],
                "Churn Score": [self.Churn_Score],
                "Tenure in Months": [self.Tenure_in_Months],
                "Monthly Charge": [self.Monthly_Charge],
                "Total Long Distance Charges": [self.Total_Long_Distance_Charges],
                "Number of Referrals": [self.Number_of_Referrals],
                "CLTV": [self.CLTV],
                "Age": [self.Age],
                "Dependents": [self.Dependents],
                "Internet Service": [self.Internet_Service],
                "Gender":[self.Gender]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)