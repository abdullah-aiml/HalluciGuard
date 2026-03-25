from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from api.detector import HallucinationDetector
from api.database import engine, Base, get_db
from api.models import HallucinationLog

detector = HallucinationDetector()

# Lifespan context to create tables automatically on startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        # In a real production app you'd use Alembic migrations, 
        # but this is perfect for our current phase.
        await conn.run_sync(Base.metadata.create_all)
    yield

app = FastAPI(
    title="HalluciGuard API",
    description="Async API for detecting LLM hallucinations using NLI.",
    version="1.0.0",
    lifespan=lifespan
)

class HallucinationRequest(BaseModel):
    context: str
    llm_output: str

@app.get("/")
async def root():
    return {"status": "online", "message": "HalluciGuard API is running."}

# Notice we added `db: AsyncSession = Depends(get_db)` here
@app.post("/api/v1/score")
async def score_hallucination(request: HallucinationRequest, db: AsyncSession = Depends(get_db)):
    
    # 1. Run the ML Model
    results = detector.analyze(request.context, request.llm_output)
    
    # 2. Package the data for PostgreSQL
    new_log = HallucinationLog(
        context=request.context,
        llm_output=request.llm_output,
        contradiction_score=results["contradiction_score"],
        entailment_score=results["entailment_score"],
        neutral_score=results["neutral_score"],
        is_hallucination=results["is_hallucination"]
    )
    
    # 3. Async commit to the database
    db.add(new_log)
    await db.commit()
    await db.refresh(new_log) # Grabs the auto-generated ID and Timestamp
    
    # 4. Return to the user
    return {
        "log_id": new_log.id,
        "context": request.context,
        "llm_output": request.llm_output,
        "results": results,
        "timestamp": new_log.created_at
    }