import os
import sys
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from exception import CustomException
from logger import logging

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models):
    try:
        report = {}
        for model_name, model in models.items():
            logging.info(f"Training and evaluating model: {model_name}")
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_test)
            
            # For probability metrics
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)[:, 1]
            else:
                y_prob = y_pred
                
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            try:
                roc_auc = roc_auc_score(y_test, y_prob)
            except Exception:
                roc_auc = 0.5 # fallback

            report[model_name] = {
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "roc_auc": float(roc_auc),
                "model_obj": model
            }
        return report
    except Exception as e:
        raise CustomException(e, sys)
