
import asyncio
from typing import Literal
import random
from shrike.evaluations.base_eval import BaseEval


class RandomEval(BaseEval):
    name:str
    type: Literal["shrike.evaluations.random.RandomEval"]

    async def check(self,executor)->bool:
        return await asyncio.get_event_loop().run_in_executor(executor, random.choice, [True, False] )