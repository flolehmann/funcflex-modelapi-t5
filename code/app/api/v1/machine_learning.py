from fastapi import Depends, APIRouter, HTTPException

import methods
import schema.prediction
from starlette import status

from definitions import MODEL_DIR

from transformers import pipeline

router = APIRouter()

ENTITY = "Machine Learning"


# load model only once:
# generator = pipeline('text-generation', model='EleutherAI/gpt-neo-1.3B', device=0)
generator = pipeline('text-generation', model='EleutherAI/gpt-neo-1.3B')

@router.post("/predict", response_model=schema.prediction.PredictionOutput,
             dependencies=[Depends(methods.api_key_authentication)])
async def predict(data: schema.prediction.PredictionInput):
    preprocess_text = data.input.strip().replace("\n", "")

    output = generator(preprocess_text, do_sample=True, min_length=200)
    result = output[0]["generated_text"].replace("\n", "")
    return {
        "prediction": result,
        "function": "generation"
    }


@router.get("/ping", status_code=status.HTTP_204_NO_CONTENT)
async def ping():
    return
