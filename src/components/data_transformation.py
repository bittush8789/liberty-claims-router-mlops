import os
import sys
import pandas as pd
from sklearn.preprocessing import StandardScaler
# Add src to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from exception import CustomException
from logger import logging
from utils import save_object

class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            scaler = StandardScaler()
            return scaler
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        logging.info("Initiated data transformation component")
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("Splitting inputs and targets")
            target_column = "Outcome"
            
            X_train = train_df.drop(columns=[target_column], axis=1)
            y_train = train_df[target_column]
            
            X_test = test_df.drop(columns=[target_column], axis=1)
            y_test = test_df[target_column]
            
            logging.info("Applying preprocessor standard scaler object")
            scaler = self.get_data_transformer_object()
            
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Save scaler
            save_object(
                file_path=self.transformation_config.preprocessor_obj_file_path,
                obj=scaler
            )
            
            logging.info("Data preprocessor saved successfully")
            
            return X_train_scaled, y_train, X_test_scaled, y_test
        except Exception as e:
            raise CustomException(e, sys)
