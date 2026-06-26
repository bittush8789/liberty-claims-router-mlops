import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import mlflow
import mlflow.sklearn
from preprocessing import preprocess_priority_target
from feature_engineering import FeaturePipeline

def train_priority():
    data_path = "data/raw_claims.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError("Data file raw_claims.csv not found.")
        
    df = pd.read_csv(data_path)
    
    # Target
    y, reverse_mapping = preprocess_priority_target(df)
    
    # Load or initialize feature pipeline
    pipeline_path = "models/feature_pipeline.joblib"
    if os.path.exists(pipeline_path):
        pipeline = FeaturePipeline.load(pipeline_path)
    else:
        pipeline = FeaturePipeline()
        
    X = pipeline.prepare_priority_features(df, is_training=True)
    
    # Save targets mapping & updated pipeline
    os.makedirs("models", exist_ok=True)
    joblib.dump(reverse_mapping, "models/priority_target_mapping.joblib")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    mlflow.set_experiment("Priority_Prediction")
    with mlflow.start_run():
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"Priority Model Accuracy: {acc:.4f}")
        
        # Log to MLflow
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_metric("accuracy", acc)
        
        mlflow.sklearn.log_model(model, "priority_model")
        
        # Save locally
        joblib.dump(model, "models/priority_model.joblib")
        pipeline.save(pipeline_path)
        
    print("Model 2 (Priority) trained and saved.")

if __name__ == "__main__":
    train_priority()
