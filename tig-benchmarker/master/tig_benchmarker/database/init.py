from sqlalchemy import create_engine, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError
import os
import logging

logger = logging.getLogger(os.path.splitext(os.path.basename(__file__))[0])

# Update the DATABASE_URL with your actual database credentials
DATABASE_URL = "postgresql+psycopg2://"+os.environ.get("POSTGRES_USER")+":"+os.environ.get("POSTGRES_PASSWORD")+"@pgpool:5432/"+os.environ.get("POSTGRES_DB")

engine = create_engine(
    DATABASE_URL,
    pool_size=20,
    max_overflow=0,
    pool_pre_ping=True,
    pool_recycle=3600,
)

SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

def execute_init_sql():
    """
    Executes the init.sql file located in the same directory as this script.
    """
    init_sql_path = os.path.join(os.path.dirname(__file__), 'init.sql')
    if not os.path.exists(init_sql_path):
        logger.error(f"init.sql file not found at {init_sql_path}")
        return False
    try:
        with open(init_sql_path, 'r') as f:
            sql = f.read()
        with engine.begin() as connection:
            connection.execute(text(sql))
        logger.info("Successfully executed init.sql")
        return True
    except SQLAlchemyError as e:
        logger.error(f"Database error while executing init.sql: {e}")
    except Exception as e:
        logger.error(f"Unexpected error while executing init.sql: {e}")
    return False