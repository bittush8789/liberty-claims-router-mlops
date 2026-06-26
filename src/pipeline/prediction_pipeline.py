import os
import sys
import pandas as pd
import numpy as np
# Add src to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from exception import CustomException
from logger import logging
from utils import load_object

class PredictionPipeline:
    def __init__(self):
        pass

    def predict(self, features: dict):
        try:
            model_path = os.path.join("models", "diabetes_model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            
            logging.info("Loading preprocessor and model objects")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            
            # Map input to dataframe
            data_df = pd.DataFrame([features])
            
            # Reorder columns to match expected schema exactly
            data_df = data_df[[
                "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
            ]]
            
            # Scale features
            scaled_features = preprocessor.transform(data_df)
            
            # Predict outcome and probability
            prediction = int(model.predict(scaled_features)[0])
            
            if hasattr(model, "predict_proba"):
                prob = float(model.predict_proba(scaled_features)[0][1])
            else:
                prob = 1.0 if prediction == 1 else 0.0
                
            return {
                "prediction": "Diabetic" if prediction == 1 else "Non-Diabetic",
                "probability": round(prob, 2)
            }
        except Exception as e:
            raise CustomException(e, sys)
