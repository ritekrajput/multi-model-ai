from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.database.models import (
    Assessment, PatientResponse, RelativeResponse, AIPrediction, ClinicalResult, RiskFlag
)
from backend.api.deps import get_db
from backend.inference.predictor import DepressionPredictor
from backend.clinical_logic.severity_mapper import map_severity
from backend.clinical_logic.risk_rules import apply_risk_rules
from backend.services.text_encoder_service import TextEncoderService

import torch
import uuid
import os

# ---- Request Models ----
class AssessmentRequest(BaseModel):
    user_id: str
    patient_text: str
    relative_text: str = None
    relationship: str = None

router = APIRouter()

# Initialize model - use relative path from project root
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "best_model.pt"
)

try:
    predictor = DepressionPredictor(model_path=MODEL_PATH)
    print(f"✅ Model loaded from {MODEL_PATH}")
except Exception as e:
    print(f"⚠️ Model loading failed: {e}")
    predictor = None

try:
    text_encoder_service = TextEncoderService(device="cpu")
    print("✅ Text encoder service initialized")
except Exception as e:
    print(f"⚠️ Text encoder service initialization failed: {e}")
    text_encoder_service = None

# ----------------------------
# Create Assessment
# ----------------------------
# ----------------------------
# Create Assessment (Unified)
# ----------------------------
@router.post("/create")
def create_assessment(
    payload: AssessmentRequest,
    db: Session = Depends(get_db)
):
    """
    Complete assessment submission in one request:
    - Create assessment
    - Save patient responses
    - Save relative responses (optional)
    - Run inference
    """
    if not predictor or not text_encoder_service:
        raise HTTPException(status_code=500, detail="Backend services not initialized")
    
    # Create assessment
    assessment = Assessment(
        id=str(uuid.uuid4()),
        patient_id=payload.user_id,  # user_id serves as patient_id
        status="pending"
    )
    db.add(assessment)
    db.commit()
    db.refresh(assessment)
    
    assessment_id = assessment.id
    
    # Save patient response as single entry
    if payload.patient_text:
        pr = PatientResponse(
            assessment_id=assessment_id,
            question_id="general",
            answer_text=payload.patient_text
        )
        db.add(pr)
    
    # Save relative response (optional)
    if payload.relative_text:
        rr = RelativeResponse(
            assessment_id=assessment_id,
            relative_id=payload.relationship or "unknown",
            question_id="general",
            answer_text=payload.relative_text
        )
        db.add(rr)
    
    db.commit()
    
    # Run inference immediately
    try:
        patient_answers = db.query(PatientResponse).filter_by(
            assessment_id=assessment_id
        ).all()
        
        relative_answers = db.query(RelativeResponse).filter_by(
            assessment_id=assessment_id
        ).all()
        
        patient_text_str = " ".join([r.answer_text for r in patient_answers])
        relative_text_str = " ".join([r.answer_text for r in relative_answers])
        
        # Encode text
        patient_emb, relative_emb = text_encoder_service.encode_patient_and_relative(
            patient_text_str,
            relative_text_str
        )
        
        # Predict
        severity = predictor.predict(
            patient_text=patient_emb,
            relative_text=relative_emb
        ).item()
        
        # Save AI prediction
        ai = AIPrediction(
            assessment_id=assessment_id,
            model_version="fusion_v1",
            severity_score=severity
        )
        db.add(ai)
        
        # Clinical logic
        clinical = map_severity(severity)
        cr = ClinicalResult(
            assessment_id=assessment_id,
            severity_level=clinical["level"],
            recommendation=clinical["action"]
        )
        db.add(cr)
        
        # Risk flags
        flags = apply_risk_rules(severity, patient_text_str, relative_text_str)
        for f in flags:
            db.add(RiskFlag(assessment_id=assessment_id, flag_type=f))
        
        # Update assessment status
        assessment.status = "completed"
        
        db.commit()
        
        return {
            "assessment_id": assessment_id,
            "severity_score": severity,
            "severity_level": clinical["level"],
            "recommendation": clinical["action"],
            "risk_flags": flags
        }
    
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


# ----------------------------
# Submit Patient Text
# ----------------------------
@router.post("/{assessment_id}/patient-text")
def submit_patient_text(
    assessment_id: str,
    responses: dict,
    db: Session = Depends(get_db)
):
    for qid, answer in responses.items():
        pr = PatientResponse(
            assessment_id=assessment_id,
            question_id=qid,
            answer_text=answer
        )
        db.add(pr)

    db.commit()
    return {"message": "Patient responses saved"}


# ----------------------------
# Submit Relative Text
# ----------------------------
@router.post("/{assessment_id}/relative-text")
def submit_relative_text(
    assessment_id: str,
    relative_id: str,
    responses: dict,
    db: Session = Depends(get_db)
):
    for qid, answer in responses.items():
        rr = RelativeResponse(
            assessment_id=assessment_id,
            relative_id=relative_id,
            question_id=qid,
            answer_text=answer
        )
        db.add(rr)

    db.commit()
    return {"message": "Relative responses saved"}


# ----------------------------
# Run AI Inference
# ----------------------------
@router.post("/{assessment_id}/run")
def run_assessment(assessment_id: str, db: Session = Depends(get_db)):
    if not predictor:
        raise HTTPException(status_code=500, detail="Model not loaded")
    if not text_encoder_service:
        raise HTTPException(status_code=500, detail="Text encoder not initialized")
    
    # ⚠️ TEMP: text embeddings mocked (replace later with real encoder)
    patient_answers = db.query(PatientResponse).filter_by(
    assessment_id=assessment_id
).all()

    relative_answers = db.query(RelativeResponse).filter_by(
    assessment_id=assessment_id
).all()

    patient_text_str = " ".join([r.answer_text for r in patient_answers])
    relative_text_str = " ".join([r.answer_text for r in relative_answers])

    patient_text, relative_text = text_encoder_service.encode_patient_and_relative(
    patient_text_str,
    relative_text_str
)

    severity = predictor.predict(
        patient_text=patient_text,
        relative_text=relative_text
    ).item()

    # Save AI prediction
    ai = AIPrediction(
        assessment_id=assessment_id,
        model_version="fusion_v1",
        severity_score=severity
    )
    db.add(ai)

    # Clinical logic
    clinical = map_severity(severity)
    cr = ClinicalResult(
        assessment_id=assessment_id,
        severity_level=clinical["level"],
        recommendation=clinical["action"]
    )
    db.add(cr)

    flags = apply_risk_rules(severity, patient_text_str, relative_text_str)
    for f in flags:
        db.add(RiskFlag(assessment_id=assessment_id, flag_type=f))

    # Update assessment status
    assessment = db.query(Assessment).get(assessment_id)
    assessment.status = "completed"

    db.commit()

    return {
        "severity_score": severity,
        "severity_level": clinical["level"],
        "recommendation": clinical["action"],
        "risk_flags": flags
    }
