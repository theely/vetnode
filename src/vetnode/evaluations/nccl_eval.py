
import asyncio
import datetime
import os
from typing import Dict, Literal

from pydantic import BaseModel


from vetnode.commands.scontrol.scontrol_command import ScontrolCommand
from vetnode.evaluations.base_eval import BaseEval
import torch
import torch.distributed as dist
from vetnode.evaluations.models import BandwithSize, BinaryByteSize


# https://stackoverflow.com/a/75332100/9201239
fmt_bytes = lambda v : str(v >> ((max(v.bit_length()-1, 0)//10)*10)) +["", "K", "M", "G", "T", "P", "E"][max(v.bit_length()-1, 0)//10]+"iB"
# following the common networking hw spec convention which uses base 10, instead of 2 for bps/Bps (it makes speed look bigger than it is)
conv_to_GBps = lambda v : v/10**9




class NCCLEvalWarmUp(BaseModel):
    payload:BinaryByteSize= '256 MB'
    runs:int= 3

class NCCLEval(BaseEval):
    name:str
    type: Literal["vetnode.evaluations.nccl_eval.NCCLEval"]
    requirements: Literal[[['torch','--index-url','https://download.pytorch.org/whl/cu126'],"numpy"]]
    scheduler:  Literal["slurm","openPBS"]
    payload: BinaryByteSize = '4 GB'
    warmup: NCCLEvalWarmUp
    min_bandwidth: BandwithSize = '15 GB/s'
    def verify(self)->bool:
        return True

    async def check(self,executor)->bool:
        return await asyncio.get_event_loop().run_in_executor(executor, self._check)


    def _check(self)->tuple[bool,dict]:

        local_rank = None
        nodes = None
        master_node = None
        match self.scheduler:
            case "slurm":
                local_rank = int(os.environ["SLURM_PROCID"])
                nodes = asyncio.run(ScontrolCommand().run()).hostnames
                master_node = nodes[0]
            case _:
                raise NotImplementedError("Support for the rquested scheduler has not been implemented.")

        dist.init_process_group(
            backend="nccl",
            init_method="tcp://{}:{}".format(master_node, 6001),
            timeout=datetime.timedelta(seconds=5),
            rank=local_rank,
            world_size=len(nodes),
        )
        torch.cuda.set_device(local_rank)
        
        tensor = None
        # /4 is for 4 bytes in fp32
        tensor = torch.rand(self.warmup.payload//4, 1, dtype=torch.float32).cuda(local_rank)
        for i in range(self.warmup.runs):
             self.timed_allreduce(local_rank,tensor,self.warmup.payload,len(nodes))

        # /4 is for 4 bytes in fp32
        tensor = torch.rand(self.payload//4, 1, dtype=torch.float32).cuda(local_rank)
        #bandwith = self.timed_allreduce(local_rank,tensor,self.payload,len(nodes))
        bandwith = self.timed_roundrobin(local_rank,tensor,self.payload,len(nodes))
        
        dist.destroy_process_group()
        
        return bandwith > self.min_bandwidth, {"bandwith":f"{conv_to_GBps(bandwith):6.2f} GB/s"}
    

    def timed_allreduce(self,local_rank,tensor,size,ranks):
        
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        dist.barrier(device_ids=[local_rank])
        start_event.record()
        dist.all_reduce(tensor)
        end_event.record()
        torch.cuda.synchronize()
        duration = start_event.elapsed_time(end_event) / 1000
        bandwith = size/duration        
        return bandwith * (2*(ranks - 1) / ranks)
    

    def timed_roundrobin(self,local_rank,tensor,size,ranks):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        for i in range(ranks):
            for j in range(ranks):

                #All processes wait here    
                dist.barrier(device_ids=[local_rank])
                
                if local_rank == i or local_rank == j:
                    start_event.record()
                    if local_rank == i:
                        print(f"#{local_rank} -> sending")
                        dist.send(tensor=tensor, dst=j)
                    else:
                        print(f"#{local_rank} -> receiving")
                        dist.recv(tensor=tensor, src=i)
                    end_event.record()
                    torch.cuda.synchronize()
                    duration = start_event.elapsed_time(end_event) / 1000
                    bandwith = size/duration 
                    print(f"From {i} to {j} bandwidth:{conv_to_GBps(bandwith):6.2f} GB/s")       
                print(f"#{local_rank} completed")
        return 0

