import mysql.connector
from mysql.connector import Error
from config import MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE


def get_connection():
    return mysql.connector.connect(
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DATABASE,
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
        cur.execute("""
            CREATE TABLE IF NOT EXISTS energy_goals (
                id INT AUTO_INCREMENT PRIMARY KEY,
                user_id INT NOT NULL,
                month_year VARCHAR(20) NOT NULL,
                target_kwh DECIMAL(10,2) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                UNIQUE KEY unique_user_month (user_id, month_year)
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
            (name, email, hashed_password),
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
        cur.execute(
            "SELECT id, name, email, password, created_at FROM users WHERE email=%s",
            (email,),
        )
        return cur.fetchone()
    finally:
        conn.close()


def save_user_goal(user_id, month_year, target_kwh):
    """Save or update a monthly energy consumption goal for a user."""
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO energy_goals (user_id, month_year, target_kwh)
            VALUES (%s, %s, %s)
            ON DUPLICATE KEY UPDATE target_kwh = VALUES(target_kwh)
            """,
            (int(user_id), str(month_year), float(target_kwh)),
        )
        conn.commit()
        return True, "Energy saving goal saved successfully."
    except Error as exc:
        return False, f"Database error while saving goal: {exc}"
    finally:
        conn.close()


def get_user_goal(user_id, month_year):
    """Retrieve an energy goal for a user for a specific month."""
    conn = get_connection()
    try:
        cur = conn.cursor(dictionary=True)
        cur.execute(
            """
            SELECT id, user_id, month_year, target_kwh, created_at
            FROM energy_goals
            WHERE user_id = %s AND month_year = %s
            """,
            (int(user_id), str(month_year)),
        )
        return cur.fetchone()
    except Error:
        return None
    finally:
        conn.close()


def get_user_goals(user_id):
    """Retrieve all energy goals set by a user, ordered by most recent."""
    conn = get_connection()
    try:
        cur = conn.cursor(dictionary=True)
        cur.execute(
            """
            SELECT id, user_id, month_year, target_kwh, created_at
            FROM energy_goals
            WHERE user_id = %s
            ORDER BY created_at DESC
            """,
            (int(user_id),),
        )
        return cur.fetchall()
    except Error:
        return []
    finally:
        conn.close()


def delete_user_goal(user_id, month_year):
    """Delete a specific goal for a user."""
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            "DELETE FROM energy_goals WHERE user_id = %s AND month_year = %s",
            (int(user_id), str(month_year)),
        )
        conn.commit()
        return True, "Goal deleted."
    except Error as exc:
        return False, f"Database error: {exc}"
    finally:
        conn.close()
