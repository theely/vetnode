


from typing import List, Optional
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
        'bps': 1 / 8,
        'kbps': 10**3 / 8,
        'mbps': 10**6 / 8,
        'gbps': 10**9 / 8,
        'tbps': 10**12 / 8,
        'pbps': 10**15 / 8,
        'ebps': 10**18 / 8,
    }