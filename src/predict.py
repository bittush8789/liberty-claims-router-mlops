import os
import joblib
import pandas as pd
import numpy as np
from preprocessing import clean_text
from feature_engineering import FeaturePipeline

class ClaimPredictor:
    def __init__(self, models_dir="models"):
        self.models_dir = models_dir
        self.pipeline = FeaturePipeline.load(os.path.join(models_dir, "feature_pipeline.joblib"))
        
        # Load models
        self.model_claim_type = joblib.load(os.path.join(models_dir, "claim_type_model.joblib"))
        self.model_priority = joblib.load(os.path.join(models_dir, "priority_model.joblib"))
        self.model_fraud = joblib.load(os.path.join(models_dir, "fraud_model.joblib"))
        self.model_routing = joblib.load(os.path.join(models_dir, "routing_model.joblib"))
        
        # Load target mappings
        self.claim_type_mapping = joblib.load(os.path.join(models_dir, "claim_type_target_mapping.joblib"))
        self.priority_mapping = joblib.load(os.path.join(models_dir, "priority_target_mapping.joblib"))
        self.routing_mapping = joblib.load(os.path.join(models_dir, "routing_target_mapping.joblib"))

    def predict(self, claim_data: dict) -> dict:
        """
        Runs the full inference pipeline for a single claim dictionary.
        """
        # 1. Claim Type Classification
        loss_desc_cleaned = clean_text(claim_data.get("loss_description", ""))
        tfidf_feats = self.pipeline.transform_claim_type_features([loss_desc_cleaned])
        
        type_probs = self.model_claim_type.predict_proba(tfidf_feats)[0]
        type_pred_idx = np.argmax(type_probs)
        claim_type = self.claim_type_mapping[type_pred_idx]
        type_confidence = float(type_probs[type_pred_idx])
        
        # 2. Priority Prediction
        # Construct DataFrame for easy transformation
        priority_df = pd.DataFrame([{
            "claim_amount": float(claim_data.get("claim_amount", 0.0)),
            "police_report": int(claim_data.get("police_report", 0)),
            "injury_involved": int(claim_data.get("injury_involved", 0)),
            "witness_available": int(claim_data.get("witness_available", 0)),
            "claim_type": claim_type
        }])
        
        priority_feats = self.pipeline.prepare_priority_features(priority_df, is_training=False)
        priority_probs = self.model_priority.predict_proba(priority_feats)[0]
        priority_pred_idx = np.argmax(priority_probs)
        priority = self.priority_mapping[priority_pred_idx]
        priority_confidence = float(priority_probs[priority_pred_idx])
        
        # 3. Fraud Detection
        # Estimate days to report (default to 0 if not provided)
        days_to_report = int(claim_data.get("days_to_report", 0))
        
        fraud_df = pd.DataFrame([{
            "claim_amount": float(claim_data.get("claim_amount", 0.0)),
            "previous_claims_count": int(claim_data.get("previous_claims_count", 0)),
            "days_to_report": days_to_report,
            "witness_available": int(claim_data.get("witness_available", 0)),
            "police_report": int(claim_data.get("police_report", 0))
        }])
        
        fraud_feats = self.pipeline.prepare_fraud_features(fraud_df, is_training=False)
        fraud_score = float(self.model_fraud.predict(fraud_feats)[0])
        fraud_score = min(max(round(fraud_score, 1), 0.0), 100.0)
        
        # Categorize fraud risk
        if fraud_score > 70:
            fraud_risk_category = "High"
        elif fraud_score > 30:
            fraud_risk_category = "Medium"
        else:
            fraud_risk_category = "Low"
            
        # 4. Routing Prediction
        routing_df = pd.DataFrame([{
            "claim_type": claim_type,
            "priority": priority,
            "fraud_score": fraud_score
        }])
        
        routing_feats = self.pipeline.prepare_routing_features(routing_df, is_training=False)
        routing_probs = self.model_routing.predict_proba(routing_feats)[0]
        routing_pred_idx = np.argmax(routing_probs)
        routing_confidence = float(routing_probs[routing_pred_idx])
        assigned_team = self.routing_mapping[routing_pred_idx]
        
        # Apply business rules overrides & confidence threshold checks
        # Rule 1: High fraud risk -> SIU Fraud Investigation Team
        if fraud_score > 80:
            assigned_team = "SIU Fraud Investigation Team"
            routing_confidence = 1.0
        # Rule 2: Low confidence predictions -> Human Review Team
        elif type_confidence < 0.4 or priority_confidence < 0.4 or routing_confidence < 0.4:
            assigned_team = "Human Review Team"
            
        return {
            "claim_type": claim_type,
            "claim_type_confidence": type_confidence,
            "priority": priority,
            "priority_confidence": priority_confidence,
            "fraud_score": fraud_score,
            "fraud_risk_category": fraud_risk_category,
            "assigned_team": assigned_team,
            "routing_confidence": routing_confidence
        }
