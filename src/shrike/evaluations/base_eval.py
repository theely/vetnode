
from abc import abstractmethod
from pydantic import BaseModel
import asyncio
import time
from shrike.evaluations.models import Evaluation


TIMEOUT = 5000

class BaseEval(BaseModel):
    name:str
    type:str


    def validate(self)->bool:
        pass

    def setup(self)->bool:
        pass     
    
    @abstractmethod
    async def check(self)->bool:
        pass
    
    async def eval(self)->Evaluation:
        result:Evaluation = Evaluation(**{"test_name":self.name, "test_type": self.type, "elapsedtime":0, "passed":False})
        start_time = time.time()
        async with asyncio.timeout(TIMEOUT):
            result.passed = self.check()
        end_time = time.time()
        result.elapsedtime = start_time-end_time
        return result
