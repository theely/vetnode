


import re
from typing import Dict, List, Optional, Literal
from pydantic import BaseModel, ByteSize

class EvalConfiguration(BaseModel, extra='allow'):
    name:str
    type:str
    requirements:Optional[List[str | List[str]]]=None

class Evaluation(BaseModel):
   test_name:str
   test_type:str
   passed:bool
   elapsedtime:int
   metrics:Optional[Dict[str,str]]=None

class EvalContext(BaseModel):
    eval_id:int=None
    nodes:Optional[List[str]]=None
    local_rank:Optional[int]=None
    rank:Optional[int]=None
    world_size:Optional[int]=None
    nodes_count:Optional[int]=None
    tasks_per_node:Optional[int]=None
    master_addr:Optional[str]=None
    master_port:Optional[int]=None
    scheduler:Literal["slurm", "standalone"]="slurm"

class BinaryByteSize(ByteSize):
    byte_sizes = {
        'b': 1,
        'kb': 2**10,
        'mb': 2**20,
        'gb': 2**30,
        'tb': 2**40,
        'pb': 2**50,
        'eb': 2**60,
    }

class BandwidthSize(ByteSize):
    byte_sizes = {
        'b/s': 1,
        'kb/s': 2**10,
        'mb/s': 2**20,
        'gb/s': 2**30,
        'tb/s': 2**40,
        'pb/s': 2**50,
        'eb/s': 2**60,
    }
    byte_string_pattern = r'^\s*(\d*\.?\d+)\s*([\w\/]+)?'
    byte_string_re = re.compile(byte_string_pattern, re.IGNORECASE)