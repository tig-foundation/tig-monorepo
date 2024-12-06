import os
import logging
import psycopg2
from psycopg2.extras import RealDictCursor
from typing import Optional, Dict, Any, List

logger = logging.getLogger(os.path.splitext(os.path.basename(__file__))[0])

class PostgresDB:
    def __init__(self, host: str, port: int, dbname: str, user: str, password: str):
        self.conn_params = {
            'host': host,
            'port': port,
            'dbname': dbname,
            'user': user,
            'password': password
        }
        self._conn = None

    def connect(self) -> None:
        """Establish connection to PostgreSQL database"""
        try:
            self._conn = psycopg2.connect(**self.conn_params)
            logger.info(f"Connected to PostgreSQL database at {self.conn_params['host']}:{self.conn_params['port']}")
        except Exception as e:
            logger.error(f"Error connecting to PostgreSQL: {str(e)}")
            raise

    def disconnect(self) -> None:
        """Close database connection"""
        if self._conn:
            self._conn.close()
            self._conn = None
            logger.info("Disconnected from PostgreSQL database")

    def execute_many(self, *args) -> None:
        """Execute multiple queries in a single transaction"""
        if not self._conn:
            self.connect()

        try:
            with self._conn.cursor() as cur:
                cur.execute("BEGIN")
                for query in args:
                    cur.execute(*query)
            self._conn.commit()
        except Exception as e:
            self._conn.rollback()
            logger.error(f"Error executing queries: {str(e)}")
            raise

    def execute(self, query: str, params: Optional[tuple] = None) -> None:
        """Execute a query without returning results"""
        if not self._conn:
            self.connect()
        
        try:
            with self._conn.cursor() as cur:
                cur.execute(query, params)
            self._conn.commit()
        except Exception as e:
            self._conn.rollback()
            logger.error(f"Error executing query: {str(e)}")
            raise

    def fetch_one(self, query: str, params: Optional[tuple] = None) -> Optional[Dict[str, Any]]:
        """Execute query and return single row as dictionary"""
        if not self._conn:
            self.connect()

        try:
            with self._conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, params)
                return cur.fetchone()
        except Exception as e:
            self._conn.rollback()
            logger.error(f"Error fetching row: {str(e)}")
            raise

    def fetch_all(self, query: str, params: Optional[tuple] = None) -> List[Dict[str, Any]]:
        """Execute query and return all rows as list of dictionaries"""
        if not self._conn:
            self.connect()

        try:
            with self._conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, params)
                return cur.fetchall()
        except Exception as e:
            self._conn.rollback() 
            logger.error(f"Error fetching rows: {str(e)}")
            raise

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()


db_conn = None
if db_conn is None:
    db_conn = PostgresDB(
        host=os.environ["POSTGRES_HOST"],
        port=5432,
        dbname=os.environ["POSTGRES_DB"],
        user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"]
    )