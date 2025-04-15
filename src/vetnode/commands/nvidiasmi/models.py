from typing import List
from vetnode.models import CamelModel


class GPUInfo(CamelModel):
    id:str
    module_id:int
    model:str
    temp:int
    memory_total:int = None
    memory_used:int = None              

class NvidiaSMIOutput(CamelModel):
   gpus:List[GPUInfo]




