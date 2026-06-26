import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import mlflow
import mlflow.sklearn
from feature_engineering import FeaturePipeline

def train_fraud():
    data_path = "data/raw_claims.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError("Data file raw_claims.csv not found.")
        
    df = pd.read_csv(data_path)
    
    # Target
    y = df["fraud_score"]
    
    # Load or initialize pipeline
    pipeline_path = "models/feature_pipeline.joblib"
    if os.path.exists(pipeline_path):
        pipeline = FeaturePipeline.load(pipeline_path)
    else:
        pipeline = FeaturePipeline()
        
    X = pipeline.prepare_fraud_features(df, is_training=True)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    mlflow.set_experiment("Fraud_Detection")
    with mlflow.start_run():
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        print(f"Fraud Model MAE: {mae:.4f}, R2 Score: {r2:.4f}")
        
        # Log to MLflow
        mlflow.log_param("model_type", "RandomForestRegressor")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        
        mlflow.sklearn.log_model(model, "fraud_model")
        
        # Save locally
        joblib.dump(model, "models/fraud_model.joblib")
        pipeline.save(pipeline_path)
        
    print("Model 3 (Fraud) trained and saved.")

if __name__ == "__main__":
    train_fraud()
