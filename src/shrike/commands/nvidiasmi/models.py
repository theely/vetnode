from typing import List, Optional
from pydantic import BaseModel
                  
class GPUInfo(BaseModel):
    id:str
    module_id:Optional[int]
    model:str
    temp:int
    memory_total:Optional[int]
    memory_used:Optional[int]               

class NvidiaSMIOutput(BaseModel):
   gpus:List[GPUInfo]