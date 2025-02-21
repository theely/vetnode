from typing import List
from pydantic import BaseModel
                  
class GPUInfo(BaseModel):
    id:str
    module_id:int
    model:str
    temp:int
    memory_total:int
    memory_used:int               

class NvidiaSMIOutput(BaseModel):
   gpus:List[GPUInfo]