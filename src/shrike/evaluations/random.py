
from typing import Literal
import random
from shrike.evaluations.base_eval import BaseEval


class RandomEval(BaseEval):
    name:str
    type: Literal["random"]

    async def check(self)->bool:
        return random.choice([True, False])