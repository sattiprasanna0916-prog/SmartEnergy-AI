import mysql.connector
from mysql.connector import Error
from config import MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE
def get_connection():
    return mysql.connector.connect(
        host=MYSQL_HOST, port=MYSQL_PORT, user=MYSQL_USER,
        password=MYSQL_PASSWORD, database=MYSQL_DATABASE
    )
def initialize_database():
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(100) NOT NULL,
                email VARCHAR(255) NOT NULL UNIQUE,
                password VARCHAR(255) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
    finally:
        conn.close()

def create_user(name, email, hashed_password):
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO users (name, email, password) VALUES (%s, %s, %s)",
            (name, email, hashed_password)
        )
        conn.commit()
        return True, "Account created successfully."
    except Error as exc:
        if getattr(exc, "errno", None) == 1062:
            return False, "This email is already registered."
        return False, f"Database error: {exc}"
    finally:
        conn.close()

def get_user_by_email(email):
    conn = get_connection()
    try:
        cur = conn.cursor(dictionary=True)
        cur.execute("SELECT id, name, email, password, created_at FROM users WHERE email=%s", (email,))
        return cur.fetchone()
    finally:
        conn.close()
