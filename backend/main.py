from fastapi import FastAPI
from backend.api.assessment import router as assessment_router
from backend.api.result import router as results_router
from backend.database.init_db import init_db


app = FastAPI(
    title="Depression Detection API",
    description="Backend for Multimodal Depression Detection System",
    version="1.0.0"
)

app.include_router(assessment_router, prefix="/assessments", tags=["Assessments"])
app.include_router(results_router, prefix="/results", tags=["Results"])

@app.on_event("startup")
def startup_event():
    """Initialize database on startup"""
    init_db()
    print("🚀 Backend started - Database initialized")

@app.get("/")
def root():
    return {"message": "Depression Detection API is running"}
