from fastapi import APIRouter, HTTPException
from auth.jwt import create_access_token
from user.models import get_user_by_email, verify_password

router = APIRouter()


@router.post("/login")
def login_user(email: str, password: str):
    """
    Authenticates a user with email and password.
    Returns a JWT access token on success.
    """
    user = get_user_by_email(email)
    if not user or not verify_password(password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    access_token = create_access_token(user_id=user.id)
    return {"access_token": access_token, "token_type": "bearer"}


@router.post("/register")
def register_user(email: str, password: str):
    """
    Creates a new user account.
    """
    from user.models import create_user
    user = create_user(email=email, password=password)
    return {"id": user.id, "email": user.email}
