import secrets
from services.email_service import send_password_reset_email

_RESET_TOKENS = {}


def generate_reset_token(email: str) -> str:
    """
    Generates a one-time password reset token for the given email
    and sends it via the email service.
    """
    token = secrets.token_urlsafe(16)
    _RESET_TOKENS[token] = email
    send_password_reset_email(email, token)
    return token


def validate_reset_token(token: str) -> str:
    """
    Validates a reset token and returns the associated email, if valid.
    """
    email = _RESET_TOKENS.get(token)
    if not email:
        raise ValueError("Invalid or expired reset token")
    return email
