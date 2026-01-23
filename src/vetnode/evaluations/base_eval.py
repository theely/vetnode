from concurrent.futures import ThreadPoolExecutor
from abc import abstractmethod
import asyncio
import time
from vetnode.evaluations.models import EvalConfiguration, Evaluation, EvalContext



TIMEOUT = 5000
_POOL = ThreadPoolExecutor(max_workers=10)


class BaseEval(EvalConfiguration):

    def __init__(self, context:EvalContext=None, **data) -> None:
        super().__init__(**data)
        self.context = context

    def verify(self)->bool:
        return True
    
    @abstractmethod
    async def check(self, executor:ThreadPoolExecutor)->tuple[bool,dict]:
        pass
    
    async def eval(self)->Evaluation:
        result:Evaluation = Evaluation(**{"test_name":self.name, "test_type": self.type, "elapsedtime":0, "passed":False, "metrics":None})
        
        #async with asyncio.timeout(TIMEOUT):
        start_time = time.time()
        result.passed, result.metrics = await self.check(_POOL)
        end_time = time.time()
        result.elapsedtime = end_time-start_time
        return result
