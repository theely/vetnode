from typing import List, Optional
from pydantic import BaseModel
                  
class GPUInfo(BaseModel):
    id:str
    module_id:Optional[int] = None
    model:str
    temp:int
    memory_total:Optional[int] = None
    memory_used:Optional[int] = None              

class NvidiaSMIOutput(BaseModel):
   gpus:List[GPUInfo]