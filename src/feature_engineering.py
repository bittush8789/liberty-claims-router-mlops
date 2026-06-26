import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib
import os

class FeaturePipeline:
    def __init__(self):
        self.tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
        self.priority_scaler = StandardScaler()
        self.fraud_scaler = StandardScaler()
        self.routing_scaler = StandardScaler()
        
    def fit_claim_type_features(self, loss_descriptions):
        return self.tfidf.fit_transform(loss_descriptions)
        
    def transform_claim_type_features(self, loss_descriptions):
        return self.tfidf.transform(loss_descriptions)
        
    def prepare_priority_features(self, df, is_training=True):
        # Input features: Claim Amount, Police Report, Injury, Witness, Claim Type
        # Map claim_type to numeric or one-hot encode
        claim_type_mapped = df["claim_type"].map({
            "Auto": 0, "Property": 1, "General Liability": 2, "Workers Compensation": 3
        }).fillna(0)
        
        features = np.column_stack([
            df["claim_amount"],
            df["police_report"],
            df["injury_involved"],
            df["witness_available"],
            claim_type_mapped
        ])
        
        if is_training:
            features = self.priority_scaler.fit_transform(features)
        else:
            features = self.priority_scaler.transform(features)
        return features

    def prepare_fraud_features(self, df, is_training=True):
        # Input: Claim Amount, Previous Claims, Days To Report, Witness, Police Report
        features = np.column_stack([
            df["claim_amount"],
            df["previous_claims_count"],
            df["days_to_report"],
            df["witness_available"],
            df["police_report"]
        ])
        
        if is_training:
            features = self.fraud_scaler.fit_transform(features)
        else:
            features = self.fraud_scaler.transform(features)
        return features

    def prepare_routing_features(self, df, is_training=True):
        # Input: Claim Type, Priority, Fraud Risk (score)
        claim_type_mapped = df["claim_type"].map({
            "Auto": 0, "Property": 1, "General Liability": 2, "Workers Compensation": 3
        }).fillna(0)
        
        priority_mapped = df["priority"].map({
            "Low": 0, "Medium": 1, "High": 2, "Critical": 3
        }).fillna(0)
        
        features = np.column_stack([
            claim_type_mapped,
            priority_mapped,
            df["fraud_score"]
        ])
        
        if is_training:
            features = self.routing_scaler.fit_transform(features)
        else:
            features = self.routing_scaler.transform(features)
        return features

    def save(self, filepath="models/feature_pipeline.joblib"):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)
        print(f"Feature Pipeline saved to {filepath}")

    @staticmethod
    def load(filepath="models/feature_pipeline.joblib"):
        return joblib.load(filepath)
