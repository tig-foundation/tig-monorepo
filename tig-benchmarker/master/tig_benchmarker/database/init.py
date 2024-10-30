from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

# Update the DATABASE_URL with your actual database credentials
DATABASE_URL = "postgresql+psycopg2://"+os.environ.get("POSTGRES_USER")+":"+os.environ.get("POSTGRES_PASSWORD")+"@localhost:5432/"+os.environ.get("POSTGRES_DB")

engine = create_engine(
    DATABASE_URL,
    pool_size=20,
    max_overflow=0,
    pool_pre_ping=True,
    pool_recycle=3600,
)

SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()