
from typing import Literal
import random
from shrike.evaluations.base_eval import BaseEval
from shrike.evaluations.models import Evaluation


class RandomEval(BaseEval):
    name:str
    type: Literal["random"]

    async def check(self)->Evaluation:
        eval:Evaluation = super().eval()
        eval.passed = random.choice([True, False])
        return eval