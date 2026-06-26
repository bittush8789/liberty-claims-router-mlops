import os
import sys
import pandas as pd
import yaml
# Add src to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from exception import CustomException
from logger import logging

class DataValidation:
    def __init__(self, raw_data_path="artifacts/raw.csv", report_path="validation_report.yaml"):
        self.raw_data_path = raw_data_path
        self.report_path = report_path
        self.expected_columns = [
            "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
        ]

    def validate(self):
        logging.info("Initiating data validation checks")
        try:
            if not os.path.exists(self.raw_data_path):
                raise FileNotFoundError(f"Raw data file {self.raw_data_path} not found.")

            df = pd.read_csv(self.raw_data_path)
            
            # 1. Missing values check
            missing_values = df.isnull().sum().to_dict()
            has_missing = any(val > 0 for val in missing_values.values())
            
            # 2. Duplicate check
            duplicate_count = int(df.duplicated().sum())
            
            # 3. Schema check (Columns mismatch)
            current_columns = df.columns.tolist()
            mismatched_columns = [col for col in self.expected_columns if col not in current_columns]
            schema_valid = len(mismatched_columns) == 0
            
            # 4. Data types validation
            data_types = {col: str(dtype) for col, dtype in df.dtypes.to_dict().items()}
            
            # Construct validation status
            validation_status = {
                "dataset_path": self.raw_data_path,
                "schema_validation": {
                    "valid": schema_valid,
                    "expected_columns": self.expected_columns,
                    "mismatched_columns": mismatched_columns
                },
                "missing_values": {
                    "has_missing": has_missing,
                    "details": {str(k): int(v) for k, v in missing_values.items()}
                },
                "duplicates": {
                    "duplicate_count": duplicate_count
                },
                "data_types": data_types,
                "overall_status": "Passed" if (schema_valid and not has_missing) else "Failed"
            }
            
            # Save validation report
            with open(self.report_path, "w") as file:
                yaml.dump(validation_status, file, default_flow_style=False)
                
            logging.info(f"Data validation report written to {self.report_path}")
            return validation_status
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    validator = DataValidation()
    validator.validate()
