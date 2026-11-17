import re
import bcrypt
from utils.database import create_user, get_user_by_email
def register_user(name, email, password):
    name, email = name.strip(), email.strip().lower()
    if not name or not email or not password:
        return False, "All fields are required."
    if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email):
        return False, "Enter a valid email address."
    if len(password) < 8:
        return False, "Password must contain at least 8 characters."
    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    return create_user(name, email, hashed)
def login_user(email, password):
    user = get_user_by_email(email.strip().lower())
    if not user or not bcrypt.checkpw(password.encode(), user["password"].encode()):
        return False, None, "Invalid email or password."
    return True, user, "Login successful."
