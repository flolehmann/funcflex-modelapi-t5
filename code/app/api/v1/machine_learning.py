from fastapi import Depends, APIRouter, HTTPException

import methods
import schema.prediction
from starlette import status

from api import utils
from definitions import MODEL_DIR

import os
import shutil

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text

router = APIRouter()

ENTITY = "Machine Learning"


@router.get("/test", status_code=status.HTTP_204_NO_CONTENT)
async def test():
    utils.test(MODEL_DIR)
    return


@router.get("/train", status_code=status.HTTP_204_NO_CONTENT)
async def train():
    return
    #BertExaample
    #bert.read()


@router.post("/predict", response_model=schema.prediction.PredictionOutput,
             dependencies=[Depends(methods.api_key_authentication)])
async def predict(data: schema.prediction.PredictionInput):
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string)
    preprocessor = hub.KerasLayer(os.path.join(MODEL_DIR, "bert", "preprocess"))
    encoder_inputs = preprocessor(text_input)
    encoder = hub.KerasLayer(os.path.join(MODEL_DIR, "bert", "encoder"), trainable=True)
    outputs = encoder(encoder_inputs)
    pooled_output = outputs["pooled_output"]  # [batch_size, 256].
    sequence_output = outputs["sequence_output"]  # [batch_size, seq_length, 256].

    print(pooled_output)
    print(sequence_output)

    return
    #return schema.prediction.PredictionOutput(prediction=results)


@router.get("/ping", status_code=status.HTTP_204_NO_CONTENT)
async def ping():
    return
