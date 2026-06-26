import os
import sys
import json
# Add src to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from exception import CustomException
from logger import logging

class ModelEvaluation:
    def __init__(self, metrics_path="metrics.json"):
        self.metrics_path = metrics_path

    def save_metrics(self, model_name, metrics):
        logging.info("Saving evaluation metrics to JSON file")
        try:
            eval_metrics = {
                "best_model": model_name,
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "roc_auc": metrics["roc_auc"]
            }
            
            with open(self.metrics_path, "w") as file:
                json.dump(eval_metrics, file, indent=4)
                
            logging.info(f"Evaluation metrics written to {self.metrics_path}")
            return eval_metrics
        except Exception as e:
            raise CustomException(e, sys)
