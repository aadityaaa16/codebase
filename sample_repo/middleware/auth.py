from fastapi import Request, HTTPException
from auth.jwt import decode_access_token


def verify_token(request: Request):
    """
    Middleware that checks for a valid Bearer token before
    allowing access to a protected route.
    """
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing token")

    token = auth_header.split(" ")[1]
    try:
        payload = decode_access_token(token)
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    return payload
