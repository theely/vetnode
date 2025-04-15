
import asyncio
import datetime
import os
from typing import Literal


from vetnode.commands.scontrol.scontrol_command import ScontrolCommand
from vetnode.evaluations.base_eval import BaseEval
import torch
import torch.distributed as dist



# following the common networking hw spec convention which uses base 10, instead of 2 for bps/Bps (it makes speed look bigger than it is)
conv_to_GBps = lambda v : v/10**9


class NCCLEval(BaseEval):
    name:str
    type: Literal["vetnode.evaluations.nccl_eval.NCCLEval"]
    requirements: Literal[[['torch','--index-url','https://download.pytorch.org/whl/cu126'],"numpy"]]
    scheduler:  Literal["slurm","openPBS"]

    def verify(self)->bool:
        return True

    async def check(self,executor)->bool:
        return await asyncio.get_event_loop().run_in_executor(executor, self._check)


    def _check(self)->bool:

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
        

        lower_limit = 32
        upper_limit = 32

        #lower_limit = 15
        #upper_limit = 34
        # 2**15 to 2**34 => 32KB to 16GB
        sizes = [2**x for x in range(lower_limit, upper_limit+1)]

        for size in sizes:
            # clear prev-iteration memory for cards w/ ~24GB
            tensor = None
            # /4 is for 4 bytes in fp32
            tensor = torch.rand(size//4, 1, dtype=torch.float32).cuda(local_rank)
            self.timed_allreduce(local_rank,tensor,size,len(nodes))

        
        dist.destroy_process_group()
        
        return True
    

    def timed_allreduce(self,local_rank,tensor,size,ranks):
        
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        dist.barrier(device_ids=[local_rank])
        start_event.record()
        dist.all_reduce(tensor)
        end_event.record()
        torch.cuda.synchronize()
        duration = start_event.elapsed_time(end_event) / 1000
        print(f"Duration: {start_event.elapsed_time(end_event)}")
        bandwith = size/duration * (2*(ranks - 1) / ranks)
        print(f" {conv_to_GBps(bandwith):6.2f}GBps")
        return True

        