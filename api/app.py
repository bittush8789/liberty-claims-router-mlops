import os
import sys
from datetime import datetime
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

# Add src to python path to import predict.py
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from predict import ClaimPredictor

# Database Configuration
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./claims.db")
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# SQLAlchemy Models
class ClaimDB(Base):
    __tablename__ = "claims"
    
    claim_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    policy_number = Column(String, index=True, nullable=False)
    claim_number = Column(String, unique=True, index=True, nullable=True)
    customer_name = Column(String, nullable=False)
    email = Column(String, nullable=True)
    phone = Column(String, nullable=True)
    date_of_loss = Column(String, nullable=True)
    state = Column(String, nullable=True)
    claim_amount = Column(Float, nullable=False)
    loss_description = Column(String, nullable=False)
    police_report = Column(Integer, default=0)
    witness_available = Column(Integer, default=0)
    injury_involved = Column(Integer, default=0)
    previous_claims_count = Column(Integer, default=0)
    days_to_report = Column(Integer, default=0)
    claim_type = Column(String, nullable=True)
    priority = Column(String, nullable=True)
    fraud_score = Column(Float, nullable=True)
    assigned_team = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class PredictionLogDB(Base):
    __tablename__ = "prediction_logs"
    
    prediction_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    claim_id = Column(Integer, ForeignKey("claims.claim_id"), nullable=True)
    model_name = Column(String, nullable=False)
    prediction = Column(String, nullable=False)
    confidence = Column(Float, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)

# Create tables
Base.metadata.create_all(bind=engine)

# Pydantic Schemas
class ClaimCreate(BaseModel):
    policy_number: str
    claim_number: Optional[str] = None
    customer_name: str
    email: Optional[EmailStr] = None
    phone: Optional[str] = None
    date_of_loss: Optional[str] = None
    state: Optional[str] = None
    claim_amount: float
    loss_description: str
    police_report: bool = False
    witness_available: bool = False
    injury_involved: bool = False
    previous_claims_count: int = 0
    days_to_report: int = 0

class ClaimResponse(BaseModel):
    claim_id: int
    policy_number: str
    claim_number: Optional[str]
    customer_name: str
    claim_amount: float
    loss_description: str
    claim_type: Optional[str]
    priority: Optional[str]
    fraud_score: Optional[float]
    assigned_team: Optional[str]
    created_at: datetime
    
    class Config:
        from_attributes = True

class PredictRequest(BaseModel):
    claim_amount: float
    loss_description: str
    police_report: bool = False
    witness_available: bool = False
    injury_involved: bool = False
    previous_claims_count: int = 0
    days_to_report: int = 0

from fastapi.responses import FileResponse

app = FastAPI(title="Liberty Claims Router API", version="1.0.0")

@app.get("/")
def read_root():
    frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend", "index.html")
    return FileResponse(frontend_path)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency to get db session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Predictor instance
predictor = None
def get_predictor():
    global predictor
    if predictor is None:
        models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
        try:
            predictor = ClaimPredictor(models_dir=models_dir)
        except Exception as e:
            # Fallback or log error
            print(f"Error loading models from {models_dir}: {e}")
            predictor = None
    return predictor

@app.post("/predict")
def predict_claim(payload: PredictRequest):
    pred_engine = get_predictor()
    if pred_engine is None:
        raise HTTPException(status_code=503, detail="Machine Learning models are not trained or loaded yet.")
    
    # Format input for predictor
    claim_data = {
        "claim_amount": payload.claim_amount,
        "loss_description": payload.loss_description,
        "police_report": 1 if payload.police_report else 0,
        "witness_available": 1 if payload.witness_available else 0,
        "injury_involved": 1 if payload.injury_involved else 0,
        "previous_claims_count": payload.previous_claims_count,
        "days_to_report": payload.days_to_report
    }
    
    try:
        results = pred_engine.predict(claim_data)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/claims", response_model=ClaimResponse)
def create_claim(payload: ClaimCreate, db: Session = Depends(get_db)):
    pred_engine = get_predictor()
    
    # Auto-generate claim number if not provided
    claim_num = payload.claim_number or f"CLM-{int(datetime.utcnow().timestamp())}"
    
    # Predict details
    if pred_engine is not None:
        claim_data = {
            "claim_amount": payload.claim_amount,
            "loss_description": payload.loss_description,
            "police_report": 1 if payload.police_report else 0,
            "witness_available": 1 if payload.witness_available else 0,
            "injury_involved": 1 if payload.injury_involved else 0,
            "previous_claims_count": payload.previous_claims_count,
            "days_to_report": payload.days_to_report
        }
        try:
            pred_results = pred_engine.predict(claim_data)
            claim_type = pred_results["claim_type"]
            priority = pred_results["priority"]
            fraud_score = pred_results["fraud_score"]
            assigned_team = pred_results["assigned_team"]
            
            type_conf = pred_results["claim_type_confidence"]
            priority_conf = pred_results["priority_confidence"]
            routing_conf = pred_results["routing_confidence"]
        except Exception as e:
            print(f"Prediction failed inside claims: {e}")
            claim_type, priority, fraud_score, assigned_team = "Human Review Team", "Medium", 50.0, "Human Review Team"
            type_conf, priority_conf, routing_conf = 0.5, 0.5, 0.5
    else:
        # Defaults if model not ready
        claim_type, priority, fraud_score, assigned_team = "Human Review Team", "Medium", 50.0, "Human Review Team"
        type_conf, priority_conf, routing_conf = 0.5, 0.5, 0.5

    # Store Claim
    db_claim = ClaimDB(
        policy_number=payload.policy_number,
        claim_number=claim_num,
        customer_name=payload.customer_name,
        email=payload.email,
        phone=payload.phone,
        date_of_loss=payload.date_of_loss,
        state=payload.state,
        claim_amount=payload.claim_amount,
        loss_description=payload.loss_description,
        police_report=1 if payload.police_report else 0,
        witness_available=1 if payload.witness_available else 0,
        injury_involved=1 if payload.injury_involved else 0,
        previous_claims_count=payload.previous_claims_count,
        days_to_report=payload.days_to_report,
        claim_type=claim_type,
        priority=priority,
        fraud_score=fraud_score,
        assigned_team=assigned_team
    )
    db.add(db_claim)
    db.commit()
    db.refresh(db_claim)
    
    # Store Prediction Logs
    logs = [
        PredictionLogDB(claim_id=db_claim.claim_id, model_name="Claim Type Classifier", prediction=claim_type, confidence=type_conf),
        PredictionLogDB(claim_id=db_claim.claim_id, model_name="Priority Predictor", prediction=priority, confidence=priority_conf),
        PredictionLogDB(claim_id=db_claim.claim_id, model_name="Fraud Regressor", prediction=str(fraud_score), confidence=1.0),
        PredictionLogDB(claim_id=db_claim.claim_id, model_name="Routing Classifier", prediction=assigned_team, confidence=routing_conf)
    ]
    db.add_all(logs)
    db.commit()
    
    return db_claim

@app.get("/claims", response_model=List[ClaimResponse])
def get_claims(db: Session = Depends(get_db)):
    return db.query(ClaimDB).order_by(ClaimDB.created_at.desc()).all()

@app.get("/claims/{claim_id}", response_model=ClaimResponse)
def get_claim_details(claim_id: int, db: Session = Depends(get_db)):
    claim = db.query(ClaimDB).filter(ClaimDB.claim_id == claim_id).first()
    if not claim:
        raise HTTPException(status_code=404, detail="Claim not found")
    return claim

@app.get("/dashboard")
def get_dashboard_stats(db: Session = Depends(get_db)):
    claims = db.query(ClaimDB).all()
    total_claims = len(claims)
    
    if total_claims == 0:
        return {
            "total_claims": 0,
            "claims_by_type": {},
            "fraud_cases": 0,
            "priority_distribution": {},
            "team_routing_distribution": {},
            "average_claim_amount": 0.0
        }
        
    # Aggegations
    claims_by_type = {}
    priority_dist = {}
    team_dist = {}
    total_amount = 0.0
    fraud_cases_count = 0
    
    for c in claims:
        claims_by_type[c.claim_type] = claims_by_type.get(c.claim_type, 0) + 1
        priority_dist[c.priority] = priority_dist.get(c.priority, 0) + 1
        team_dist[c.assigned_team] = team_dist.get(c.assigned_team, 0) + 1
        total_amount += c.claim_amount
        if c.fraud_score and c.fraud_score > 70.0:
            fraud_cases_count += 1
            
    return {
        "total_claims": total_claims,
        "claims_by_type": claims_by_type,
        "fraud_cases": fraud_cases_count,
        "priority_distribution": priority_dist,
        "team_routing_distribution": team_dist,
        "average_claim_amount": round(total_amount / total_claims, 2)
    }
