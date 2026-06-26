import os
import sys
import time
from typing import List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel, Field

# Add src to python path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from pipeline.prediction_pipeline import PredictionPipeline

# Initialize FastAPI app
app = FastAPI(title="End-to-End Diabetes Prediction MLOps Platform", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus metric placeholders
metrics_store = {
    "prediction_count": 0,
    "diabetic_count": 0,
    "non_diabetic_count": 0,
    "error_count": 0,
    "latencies": []
}

class PatientData(BaseModel):
    Pregnancies: int = Field(..., ge=0, description="Number of times pregnant")
    Glucose: int = Field(..., ge=0, description="Plasma glucose concentration a 2 hours in an oral glucose tolerance test")
    BloodPressure: int = Field(..., ge=0, description="Diastolic blood pressure (mm Hg)")
    SkinThickness: int = Field(..., ge=0, description="Triceps skin fold thickness (mm)")
    Insulin: int = Field(..., ge=0, description="2-Hour serum insulin (mu U/ml)")
    BMI: float = Field(..., ge=0.0, description="Body mass index (weight in kg/(height in m)^2)")
    DiabetesPedigreeFunction: float = Field(..., ge=0.0, description="Diabetes pedigree function")
    Age: int = Field(..., ge=0, description="Age (years)")

class BatchPredictRequest(BaseModel):
    patients: List[PatientData]

@app.get("/")
def read_root():
    frontend_path = os.path.join(os.path.dirname(__file__), "frontend", "index.html")
    if not os.path.exists(frontend_path):
        return {"message": "Diabetes Prediction Platform Online. UI files not built yet."}
    return FileResponse(frontend_path)

@app.get("/health")
def health_check():
    # Verify model files exist
    model_exists = os.path.exists(os.path.join("models", "diabetes_model.pkl"))
    preprocessor_exists = os.path.exists(os.path.join("artifacts", "preprocessor.pkl"))
    
    if model_exists and preprocessor_exists:
        return {"status": "healthy", "model_loaded": True, "preprocessor_loaded": True}
    else:
        return {"status": "degraded", "model_loaded": model_exists, "preprocessor_loaded": preprocessor_exists}

@app.get("/metrics", response_class=PlainTextResponse)
def get_metrics():
    # Format metrics in Prometheus exporter standard text format
    avg_latency = sum(metrics_store["latencies"]) / len(metrics_store["latencies"]) if metrics_store["latencies"] else 0.0
    
    res = f"""# HELP diabetes_predictions_total Total count of diabetes predictions served
# TYPE diabetes_predictions_total counter
diabetes_predictions_total {metrics_store['prediction_count']}

# HELP diabetes_diabetic_total Total count of predicted diabetic cases
# TYPE diabetes_diabetic_total counter
diabetes_diabetic_total {metrics_store['diabetic_count']}

# HELP diabetes_non_diabetic_total Total count of predicted non-diabetic cases
# TYPE diabetes_non_diabetic_total counter
diabetes_non_diabetic_total {metrics_store['non_diabetic_count']}

# HELP diabetes_errors_total Total API and prediction errors encountered
# TYPE diabetes_errors_total counter
diabetes_errors_total {metrics_store['error_count']}

# HELP diabetes_prediction_latency_seconds_average Average latency for prediction requests
# TYPE diabetes_prediction_latency_seconds_average gauge
diabetes_prediction_latency_seconds_average {avg_latency:.4f}
"""
    return res

@app.post("/predict")
def predict(payload: PatientData):
    start_time = time.time()
    try:
        pipeline = PredictionPipeline()
        features = payload.model_dump()
        result = pipeline.predict(features)
        
        # Track metrics
        metrics_store["prediction_count"] += 1
        if result["prediction"] == "Diabetic":
            metrics_store["diabetic_count"] += 1
        else:
            metrics_store["non_diabetic_count"] += 1
            
        latency = time.time() - start_time
        metrics_store["latencies"].append(latency)
        
        return result
    except Exception as e:
        metrics_store["error_count"] += 1
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch-predict")
def batch_predict(payload: BatchPredictRequest):
    results = []
    pipeline = PredictionPipeline()
    for patient in payload.patients:
        try:
            res = pipeline.predict(patient.model_dump())
            results.append(res)
        except Exception as e:
            results.append({"error": str(e)})
    return {"predictions": results}
