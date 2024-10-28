import psycopg2
import time
import os
import subprocess
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def connect_db():
    host = os.getenv('PG_HOST', 'localhost')  # Pgpool is exposed on localhost
    port = int(os.getenv('PG_PORT', '5432'))  # Pgpool port
    user = os.getenv('POSTGRES_USER', 'postgres')
    password = os.getenv('POSTGRES_PASSWORD', 'your_secure_password')
    dbname = os.getenv('POSTGRES_DB', 'benchmarker')

    try:
        conn = psycopg2.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            dbname=dbname
        )
        return conn
    except psycopg2.Error as e:
        print(f"Error connecting to the database: {e}")
        raise

def create_table(conn):
    with conn.cursor() as cur:
        cur.execute('''
            CREATE TABLE IF NOT EXISTS test_table (
                id SERIAL PRIMARY KEY,
                data TEXT NOT NULL
            );
        ''')
        conn.commit()

def insert_data(conn, data):
    with conn.cursor() as cur:
        cur.execute('''
            INSERT INTO test_table (data) VALUES (%s) RETURNING id;
        ''', (data,))
        inserted_id = cur.fetchone()[0]
        conn.commit()
    return inserted_id

def read_data(conn):
    with conn.cursor() as cur:
        cur.execute('SELECT id, data FROM test_table ORDER BY id;')
        results = cur.fetchall()
    return results

def stop_master():
    print("Stopping the master database container (db)...")
    subprocess.run(['docker-compose', 'stop', 'db'])
    print("Master database container stopped.")

def start_master():
    print("Starting the master database container (db)...")
    subprocess.run(['docker-compose', 'start', 'db'])
    print("Master database container started.")

def main():
    conn = None
    conn_failover = None
    conn_reconnect = None

    try:
        print("Connecting to database via Pgpool...")
        conn = connect_db()
        print("Connection established.")

        print("Creating test table...")
        create_table(conn)
        print("Test table created.")

        print("Inserting data into test table...")
        inserted_id = insert_data(conn, 'Test data before failover')
        print(f"Data inserted with ID: {inserted_id}")

        print("Reading data from test table...")
        results = read_data(conn)
        for row in results:
            print(f"ID: {row[0]}, Data: {row[1]}")

        print("\nSimulating master failure...")
        stop_master()
        print("Waiting for Pgpool to detect the failure and promote the replica...")
        time.sleep(15)  # Wait for Pgpool to detect the failure and promote the replica

        print("\nAttempting to read data after master failure...")
        try:
            conn_failover = connect_db()
            results = read_data(conn_failover)
            for row in results:
                print(f"ID: {row[0]}, Data: {row[1]}")
            print("Read successful after master failure. Failover is working.")
        except Exception as e:
            print(f"Error reading data after master failure: {e}")

        print("\nAttempting to insert data after master failure...")
        try:
            if conn_failover is None:
                conn_failover = connect_db()
            inserted_id = insert_data(conn_failover, 'Test data after failover')
            print(f"Data inserted with ID: {inserted_id}")
            print("Write successful after master failure.")
        except Exception as e:
            print(f"Error inserting data after master failure: {e}")

        print("\nRestarting master database...")
        start_master()
        print("Waiting for the master to come back up and synchronize...")
        time.sleep(15)  # Wait for the master to synchronize

        print("\nReading data after master restart...")
        conn_reconnect = connect_db()
        results = read_data(conn_reconnect)
        for row in results:
            print(f"ID: {row[0]}, Data: {row[1]}")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        print("\nClosing database connections.")
        if conn:
            conn.close()
        if conn_failover:
            conn_failover.close()
        if conn_reconnect:
            conn_reconnect.close()
        print("Test completed.")

if __name__ == '__main__':
    main()
