import jwt
import datetime

SECRET_KEY = "dev-secret-change-me"
ALGORITHM = "HS256"


def create_access_token(user_id: int, expires_minutes: int = 30) -> str:
    """
    Generates a signed JWT access token for the given user.
    """
    payload = {
        "sub": str(user_id),
        "exp": datetime.datetime.utcnow() + datetime.timedelta(minutes=expires_minutes),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_access_token(token: str) -> dict:
    """
    Decodes and validates a JWT access token.
    Raises jwt.ExpiredSignatureError or jwt.InvalidTokenError on failure.
    """
    return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
