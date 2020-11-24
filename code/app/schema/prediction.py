from typing import Union

from pydantic import BaseModel


class PredictionBase(BaseModel):
    input: Union[dict, list, set, float, int, str, bytes, bool]


class PredictionInput(PredictionBase):
    pass


class PredictionOutput(BaseModel):
    prediction: Union[dict, list, set, float, int, str, bytes, bool]
    pass
