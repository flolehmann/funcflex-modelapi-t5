from fastapi import Depends, APIRouter, HTTPException

import methods
import schema.prediction
from starlette import status

from api import utils
from definitions import MODEL_DIR

router = APIRouter()

ENTITY = "Machine Learning"


@router.get("/test", status_code=status.HTTP_204_NO_CONTENT)
async def test():
    utils.test(MODEL_DIR)
    return


@router.post("/predict", response_model=schema.prediction.PredictionOutput,
             dependencies=[Depends(methods.api_key_authentication)])
async def predict(data: schema.prediction.PredictionInput):
    model_input = utils.preprocessing(data.input)
    results = model.predict(model_input)

    return schema.prediction.PredictionOutput(prediction=results)


@router.get("/ping", status_code=status.HTTP_204_NO_CONTENT)
async def ping():
    return
