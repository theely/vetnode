
from typing import Literal

from shrike.evaluations.base_eval import BaseEval


class PingEval(BaseEval):
    name:str
    type: Literal["ping-nodes"]

    async def check(self)->bool:
        return True