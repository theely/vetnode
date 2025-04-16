


import re
from typing import Dict, List, Optional
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
   metadata:Optional[Dict[str,str]]=None


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

class BandwithSize(ByteSize):
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