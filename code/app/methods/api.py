from decouple import config
from fastapi import Depends, HTTPException, Security
from fastapi.security import APIKeyHeader
from starlette import status
from typing import Optional

API_KEY = config("API_KEY")
PUBLIC_PREDICTION = config("PUBLIC_PREDICTION", cast=bool)

X_API_KEY = APIKeyHeader(name="X-API-Key", auto_error=False)


# TODO: x_api_key dependency should be optional (otherwise public prediction will not work)
def api_key_authentication(x_api_key: Optional[str] = Security(X_API_KEY)):
    if PUBLIC_PREDICTION:
        return True

    if API_KEY == x_api_key:
        return True

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid API key",
    )
