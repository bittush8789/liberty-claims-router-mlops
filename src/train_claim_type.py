import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import mlflow
import mlflow.sklearn
from ingestion import generate_synthetic_data
from preprocessing import clean_text, preprocess_claim_type_target
from feature_engineering import FeaturePipeline

def train_claim_type():
    # Setup data
    data_path = "data/raw_claims.csv"
    if not os.path.exists(data_path):
        generate_synthetic_data()
        
    df = pd.read_csv(data_path)
    
    # Preprocess text
    df["clean_description"] = df["loss_description"].apply(clean_text)
    
    # Targets
    y, reverse_mapping = preprocess_claim_type_target(df)
    
    # Features
    pipeline = FeaturePipeline()
    X = pipeline.fit_claim_type_features(df["clean_description"])
    
    # Save target mapping
    os.makedirs("models", exist_ok=True)
    joblib.dump(reverse_mapping, "models/claim_type_target_mapping.joblib")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    mlflow.set_experiment("Claim_Type_Classification")
    with mlflow.start_run():
        model = LogisticRegression(max_iter=1000, C=1.0)
        model.fit(X_train, y_train)
        
        # Evaluate
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"Claim Type Model Accuracy: {acc:.4f}")
        
        # Log to MLflow
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("C", 1.0)
        mlflow.log_metric("accuracy", acc)
        
        mlflow.sklearn.log_model(model, "claim_type_model")
        
        # Save locally
        joblib.dump(model, "models/claim_type_model.joblib")
        pipeline.save("models/feature_pipeline.joblib")
        
    print("Model 1 (Claim Type) trained and saved.")

if __name__ == "__main__":
    train_claim_type()
