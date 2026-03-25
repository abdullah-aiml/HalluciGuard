import os
from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import declarative_base, sessionmaker

# Load the .env file
load_dotenv()

# Grab the URL, with a fallback just in case
# --- COMMENT OUT YOUR POSTGRES URL ---
# DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://postgres:postgres@localhost:5432/halluciguard")

# --- ADD THE SQLITE URL ---
DATABASE_URL = "sqlite+aiosqlite:///./halluciguard.db"

# Update your engine to include connect_args for SQLite
engine = create_async_engine(
    DATABASE_URL, 
    echo=False, 
    connect_args={"check_same_thread": False} # Required for SQLite with FastAPI
)

# Create a session factory
AsyncSessionLocal = sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False
)

Base = declarative_base()

# Dependency to yield database sessions for our FastAPI routes
async def get_db():
    async with AsyncSessionLocal() as session:
        yield session