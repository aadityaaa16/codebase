def send_password_reset_email(email: str, reset_token: str):
    """
    Sends a password reset email containing a reset link with the token.
    In production this would call an SMTP or third-party email API.
    """
    reset_link = f"https://example.com/reset-password?token={reset_token}"
    print(f"Sending password reset email to {email}: {reset_link}")
    return True


class EmailService:
    """
    Wraps outgoing transactional email logic (verification, password reset).
    """

    def __init__(self, sender_address: str = "no-reply@example.com"):
        self.sender_address = sender_address

    def send_verification_email(self, email: str, verification_code: str):
        """
        Sends an email verification code to a newly registered user.
        """
        print(f"Sending verification code {verification_code} to {email}")
        return True
