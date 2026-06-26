import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# Add src to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from exception import CustomException
from logger import logging

class DataIngestionConfig:
    raw_data_path: str = os.path.join("artifacts", "raw.csv")
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion component")
        try:
            # We'll create a synthetic Pima Indians Diabetes Dataset
            np.random.seed(42)
            n_samples = 800
            
            pregnancies = np.random.randint(0, 15, size=n_samples)
            glucose = np.random.normal(120, 30, size=n_samples).clip(40, 200).astype(int)
            blood_pressure = np.random.normal(70, 12, size=n_samples).clip(30, 120).astype(int)
            skin_thickness = np.random.normal(20, 10, size=n_samples).clip(0, 60).astype(int)
            insulin = np.random.normal(80, 50, size=n_samples).clip(0, 600).astype(int)
            bmi = np.random.normal(32, 6, size=n_samples).clip(15, 60)
            dpf = np.random.exponential(scale=0.4, size=n_samples).clip(0.08, 2.4)
            age = np.random.normal(33, 11, size=n_samples).clip(21, 80).astype(int)
            
            # Simple probability model for outcome
            prob = (
                (pregnancies * 0.05) + 
                ((glucose - 100) * 0.015) + 
                ((blood_pressure - 70) * 0.005) + 
                ((bmi - 25) * 0.04) + 
                (dpf * 0.5) + 
                ((age - 30) * 0.01)
            )
            prob = 1 / (1 + np.exp(-prob)) # Sigmoid
            outcome = (prob > 0.45).astype(int)
            
            df = pd.DataFrame({
                "Pregnancies": pregnancies,
                "Glucose": glucose,
                "BloodPressure": blood_pressure,
                "SkinThickness": skin_thickness,
                "Insulin": insulin,
                "BMI": np.round(bmi, 1),
                "DiabetesPedigreeFunction": np.round(dpf, 3),
                "Age": age,
                "Outcome": outcome
            })
            
            logging.info("Generated synthetic Pima Indians Diabetes dataset successfully.")
            
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            
            logging.info("Split started")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            
            logging.info("Ingestion of data completed successfully")
            
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()
