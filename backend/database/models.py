from sqlalchemy import (
    Column, String, Integer, Float, Text, DateTime, Enum, ForeignKey
)
#from sqlalchemy.orm import relationship
from sqlalchemy.orm import relationship

from datetime import datetime
import uuid

from backend.database.base import Base

def uuid_str():
    return str(uuid.uuid4())


# ------------------------
# USERS
# ------------------------
class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=uuid_str)
    name = Column(String)
    email = Column(String, unique=True, index=True)
    role = Column(Enum("patient", "relative", "admin", name="user_roles"))
    created_at = Column(DateTime, default=datetime.utcnow)


# ------------------------
# PATIENTS
# ------------------------
class Patient(Base):
    __tablename__ = "patients"

    id = Column(String, primary_key=True, default=uuid_str)
    user_id = Column(String, ForeignKey("users.id"))
    age = Column(Integer)
    gender = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User")


# ------------------------
# RELATIVES
# ------------------------
class Relative(Base):
    __tablename__ = "relatives"

    id = Column(String, primary_key=True, default=uuid_str)
    user_id = Column(String, ForeignKey("users.id"))
    patient_id = Column(String, ForeignKey("patients.id"))
    relation_type = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User")
    patient = relationship("Patient")


# ------------------------
# ASSESSMENTS
# ------------------------
class Assessment(Base):
    __tablename__ = "assessments"

    id = Column(String, primary_key=True, default=uuid_str)
    patient_id = Column(String, ForeignKey("patients.id"))
    status = Column(Enum("pending", "completed", name="assessment_status"))
    created_at = Column(DateTime, default=datetime.utcnow)

    patient = relationship("Patient")


# ------------------------
# PATIENT RESPONSES
# ------------------------
class PatientResponse(Base):
    __tablename__ = "patient_responses"

    id = Column(String, primary_key=True, default=uuid_str)
    assessment_id = Column(String, ForeignKey("assessments.id"))
    question_id = Column(String)
    answer_text = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)


# ------------------------
# RELATIVE RESPONSES
# ------------------------
class RelativeResponse(Base):
    __tablename__ = "relative_responses"

    id = Column(String, primary_key=True, default=uuid_str)
    assessment_id = Column(String, ForeignKey("assessments.id"))
    relative_id = Column(String, ForeignKey("relatives.id"))
    question_id = Column(String)
    answer_text = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)


# ------------------------
# AUDIO FILES
# ------------------------
class AudioRecord(Base):
    __tablename__ = "audio_records"

    id = Column(String, primary_key=True, default=uuid_str)
    assessment_id = Column(String, ForeignKey("assessments.id"))
    file_path = Column(Text)
    duration_sec = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)


# ------------------------
# VIDEO FILES
# ------------------------
class VideoRecord(Base):
    __tablename__ = "video_records"

    id = Column(String, primary_key=True, default=uuid_str)
    assessment_id = Column(String, ForeignKey("assessments.id"))
    file_path = Column(Text)
    duration_sec = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)


# ------------------------
# AI PREDICTIONS
# ------------------------
class AIPrediction(Base):
    __tablename__ = "ai_predictions"

    id = Column(String, primary_key=True, default=uuid_str)
    assessment_id = Column(String, ForeignKey("assessments.id"))
    model_version = Column(String)
    severity_score = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)


# ------------------------
# CLINICAL RESULTS
# ------------------------
class ClinicalResult(Base):
    __tablename__ = "clinical_results"

    id = Column(String, primary_key=True, default=uuid_str)
    assessment_id = Column(String, ForeignKey("assessments.id"))
    severity_level = Column(
        Enum("Early", "Moderate", "Severe", name="severity_levels")
    )
    recommendation = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)


# ------------------------
# RISK FLAGS
# ------------------------
class RiskFlag(Base):
    __tablename__ = "risk_flags"

    id = Column(String, primary_key=True, default=uuid_str)
    assessment_id = Column(String, ForeignKey("assessments.id"))
    flag_type = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)


# ------------------------
# AUDIT LOGS
# ------------------------
class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(String, primary_key=True, default=uuid_str)
    user_id = Column(String)
    action = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)
