from fastapi import Depends, APIRouter, HTTPException

import methods
import schema.prediction
from starlette import status

from definitions import MODEL_DIR

from transformers import BertTokenizer, EncoderDecoderModel

router = APIRouter()

ENTITY = "Machine Learning"


# load model only once:
tokenizer = BertTokenizer.from_pretrained("google/bert2bert_L-24_wmt_de_en", pad_token="<pad>", eos_token="</s>", bos_token="<s>")
bert2bert = EncoderDecoderModel.from_pretrained("google/bert2bert_L-24_wmt_de_en")


@router.post("/predict", response_model=schema.prediction.PredictionOutput,
             dependencies=[Depends(methods.api_key_authentication)])
async def predict(data: schema.prediction.PredictionInput):
    preprocess_text = data.input.strip().replace("\n", "")

    input_ids = tokenizer(preprocess_text, return_tensors="pt", add_special_tokens=False).input_ids
    output_ids = bert2bert.generate(input_ids)[0]

    output = tokenizer.decode(output_ids, skip_special_tokens=True)

    return {
        "prediction": output,
        "function": "translation_de_en"
    }


@router.get("/ping", status_code=status.HTTP_204_NO_CONTENT)
async def ping():
    return
