


from typing import List, Optional
from pydantic import BaseModel

class EvalConfiguration(BaseModel, extra='allow'):
    name:str
    type:str
    requirements:Optional[List[str]]=None

class Evaluation(BaseModel):
   test_name:str
   test_type:str
   passed:bool
   elapsedtime:int