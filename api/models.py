from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime
from sqlalchemy.sql import func
from api.database import Base

class HallucinationLog(Base):
    __tablename__ = "hallucination_logs"

    id = Column(Integer, primary_key=True, index=True)
    context = Column(String, nullable=False)
    llm_output = Column(String, nullable=False)
    
    # ML Scores
    contradiction_score = Column(Float)
    entailment_score = Column(Float)
    neutral_score = Column(Float)
    is_hallucination = Column(Boolean)
    
    # Auto-generate a timestamp when the record is created
    created_at = Column(DateTime(timezone=True), server_default=func.now())