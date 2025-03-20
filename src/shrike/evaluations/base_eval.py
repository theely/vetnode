from concurrent.futures import ThreadPoolExecutor
from abc import abstractmethod
from pydantic import BaseModel
import asyncio
import time
from shrike.evaluations.models import Evaluation
import subprocess
import sys

TIMEOUT = 5000

_POOL = ThreadPoolExecutor(max_workers=10)


class BaseEval(BaseModel):
    name:str
    type:str


    def verify(self)->bool:
        return True

    def setup(self)->bool:
        pass     
    
    @abstractmethod
    async def check(self, executor:ThreadPoolExecutor)->bool:
        pass
    
    async def eval(self)->Evaluation:
        result:Evaluation = Evaluation(**{"test_name":self.name, "test_type": self.type, "elapsedtime":0, "passed":False})
        
        async with asyncio.timeout(TIMEOUT):
            start_time = time.time()
            result.passed = await self.check(_POOL)
            end_time = time.time()
            result.elapsedtime = end_time-start_time
        return result
    
    def install(self, package):
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
