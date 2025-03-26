from concurrent.futures import ThreadPoolExecutor
from abc import abstractmethod
import asyncio
import time
from vetnode.evaluations.models import EvalConfiguration, Evaluation



TIMEOUT = 5000
_POOL = ThreadPoolExecutor(max_workers=10)


class BaseEval(EvalConfiguration):

    def verify(self)->bool:
        return True
    
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
