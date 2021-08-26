from fastapi import FastAPI
import logging

from api.v1 import machine_learning

#logging.config.fileConfig("logging.conf", disable_existing_loggers=False)
#logger = logging.getLogger('study-predictor')

API_PREFIX = "/api/v1"

app = FastAPI(title="study-align-prediction-api", version="1")

app.include_router(
    machine_learning.router,
    prefix=API_PREFIX,
    tags=["Machine Learning"]
)
