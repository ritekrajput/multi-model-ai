from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.database.models import (
    AIPrediction, ClinicalResult, RiskFlag, Assessment
)
from backend.api.deps import get_db

router = APIRouter()


# ----------------------------
# Get Assessment Result
# ----------------------------
@router.get("/{assessment_id}")
def get_result(assessment_id: str, db: Session = Depends(get_db)):
    ai = db.query(AIPrediction).filter_by(assessment_id=assessment_id).first()
    clinical = db.query(ClinicalResult).filter_by(assessment_id=assessment_id).first()
    flags = db.query(RiskFlag).filter_by(assessment_id=assessment_id).all()

    return {
        "severity_score": ai.severity_score if ai else None,
        "severity_level": clinical.severity_level if clinical else None,
        "recommendation": clinical.recommendation if clinical else None,
        "risk_flags": [f.flag_type for f in flags]
    }


# ----------------------------
# Get Patient History
# ----------------------------
@router.get("/patient/{patient_id}")
def get_patient_history(patient_id: str, db: Session = Depends(get_db)):
    predictions = (
        db.query(AIPrediction)
        .join(Assessment, Assessment.id == AIPrediction.assessment_id)
        .filter(Assessment.patient_id == patient_id)
        .all()
    )

    return [
        {
            "assessment_id": p.assessment_id,
            "severity_score": p.severity_score
        }
        for p in predictions
    ]
