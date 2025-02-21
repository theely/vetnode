
from typing import Literal

from shrike.evaluations.base_eval import BaseEval
from shrike.evaluations.models import Evaluation


class PingEval(BaseEval):
    name:str
    type: Literal["ping-nodes"]

    async def eval(self)->Evaluation:
        eval:Evaluation = super().eval()
        eval.passed = True
        return eval