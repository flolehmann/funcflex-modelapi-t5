from decouple import config
from fastapi import Depends, HTTPException
from fastapi.security import APIKeyHeader
from starlette import status

API_KEY = config("API_KEY")
PUBLIC_PREDICTION = config("PUBLIC_PREDICTION")

X_API_KEY = APIKeyHeader(name="X-API-Key")

# TODO: x_api_key dependency should be optional (otherwise public prediction will not work)
def api_key_authentication(x_api_key: str = Depends(X_API_KEY)):
    if PUBLIC_PREDICTION:
        return True

    if API_KEY == x_api_key:
        return True

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid API key",
    )
