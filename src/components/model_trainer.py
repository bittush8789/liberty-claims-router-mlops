import os
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# Add src to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from exception import CustomException
from logger import logging
from utils import save_object, evaluate_models

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("models", "diabetes_model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, X_train, y_train, X_test, y_test):
        logging.info("Initiating model training step")
        try:
            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
                "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
            }
            
            if XGB_AVAILABLE:
                models["XGBoost"] = XGBClassifier(n_estimators=100, max_depth=5, random_state=42, eval_metric='logloss')
            else:
                models["XGBoost"] = GradientBoostingClassifier(n_estimators=100, random_state=42)
                
            model_report = evaluate_models(
                X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models
            )
            
            # Select best model based on accuracy
            best_model_name = None
            best_model_score = -1.0
            best_model_metrics = None
            
            for model_name, info in model_report.items():
                if info["accuracy"] > best_model_score:
                    best_model_score = info["accuracy"]
                    best_model_name = model_name
                    best_model_metrics = info
            
            best_model = best_model_metrics["model_obj"]
            
            if best_model_score < 0.6:
                raise CustomException("No suitable model found with accuracy above threshold", sys)
                
            logging.info(f"Best model selected: {best_model_name} with Accuracy: {best_model_score}")
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
            return best_model_name, best_model_score, best_model_metrics
        except Exception as e:
            raise CustomException(e, sys)
