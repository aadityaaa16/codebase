import hashlib
from dataclasses import dataclass


@dataclass
class User:
    id: int
    email: str
    hashed_password: str


_FAKE_DB = {}
_NEXT_ID = 1


def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Checks a plaintext password against a stored hash.
    """
    return _hash_password(plain_password) == hashed_password


def get_user_by_email(email: str):
    """
    Looks up a user record by email address.
    """
    return _FAKE_DB.get(email)


def create_user(email: str, password: str) -> User:
    """
    Creates and stores a new user with a hashed password.
    """
    global _NEXT_ID
    user = User(id=_NEXT_ID, email=email, hashed_password=_hash_password(password))
    _FAKE_DB[email] = user
    _NEXT_ID += 1
    return user
