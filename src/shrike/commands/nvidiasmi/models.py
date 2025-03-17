from typing import List
from pydantic import BaseModel, ConfigDict

def to_camel(string: str) -> str:
    return ''.join(word.capitalize() for word in string.split('_'))

class CamelModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        arbitrary_types_allowed=True,
        populate_by_name=True,
        validate_assignment=True,
    )


class GPUInfo(CamelModel):
    id:str
    module_id:int
    model:str
    temp:int
    memory_total:int = None
    memory_used:int = None              

class NvidiaSMIOutput(BaseModel):
   gpus:List[GPUInfo]



