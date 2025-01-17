from typing import Union, Optional

from pydantic import BaseModel


class PredictionBase(BaseModel):
    input: Union[dict, list, set, float, int, str, bytes, bool]
    function: Optional[str]


class PredictionInput(PredictionBase):
    pass


class PredictionOutput(BaseModel):
    prediction: Union[dict, list, set, float, int, str, bytes, bool]
    function: Optional[str]
    pass
