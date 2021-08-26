from fastapi import Depends, APIRouter, HTTPException

import methods
import schema.prediction
from starlette import status

from definitions import MODEL_DIR

from transformers import T5Tokenizer, T5ForConditionalGeneration


router = APIRouter()

ENTITY = "Machine Learning"


# load model only once:

# device = torch.device('cuda')
# model = T5ForConditionalGeneration.from_pretrained('t5-small').to(device)
model = T5ForConditionalGeneration.from_pretrained('t5-small')
tokenizer = T5Tokenizer.from_pretrained('t5-small')

@router.post("/predict", response_model=schema.prediction.PredictionOutput,
             dependencies=[Depends(methods.api_key_authentication)])
async def predict(data: schema.prediction.PredictionInput):
    print(data)

    #TODO: Add validator to schema
    model_function = "summarize"
    if data.function == "summarization":
        model_function = "summarize"
    elif data.function == "translation_en_to_de":
        model_function = "translate English to German"
    elif data.function == "translation_en_to_fr":
        model_function = "translate English to French"

    preprocess_text = data.input.strip().replace("\n", "")
    t5_prepared_text = model_function + ": " + preprocess_text

    #tokenized_text = tokenizer.encode(t5_prepared_Text, return_tensors="pt").to(device)
    tokenized_text = tokenizer.encode(t5_prepared_text, return_tensors="pt")

    input_ids = model.generate(tokenized_text,
                                 min_length=30,
                                 max_length=500
                               )

    output = tokenizer.decode(input_ids[0], skip_special_tokens=True)

    return {
        "prediction": output,
        "function": data.function
    }


@router.get("/ping", status_code=status.HTTP_204_NO_CONTENT)
async def ping():
    return
