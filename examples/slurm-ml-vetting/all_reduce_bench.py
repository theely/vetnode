
from pathlib import Path
import datetime
import gc
import os
import signal
import socket
import sys
import textwrap
import time
import torch
import torch.distributed as dist


WARMUPS = 5
TRIALS = 20

# https://stackoverflow.com/a/75332100/9201239
fmt_bytes = lambda v : str(v >> ((max(v.bit_length()-1, 0)//10)*10)) +["", "K", "M", "G", "T", "P", "E"][max(v.bit_length()-1, 0)//10]+"iB"
# following the common networking hw spec convention which uses base 10, instead of 2 for bps/Bps (it makes speed look bigger than it is)
conv_to_GBps = lambda v : v/10**9

def get_device_info():
    if torch.cuda.is_available():
        return repr(torch.cuda.get_device_properties('cuda'))
    else:
        return "Unknown accelerator"


def timed_allreduce(local_rank,tensor, size, start_event, end_event):
    dist.barrier(device_ids=[local_rank])
    start_event.record()
    dist.all_reduce(tensor)
    end_event.record()
    torch.cuda.synchronize()
    duration = start_event.elapsed_time(end_event) / 1000

    n = dist.get_world_size()
    # note that this is following the same math as NVIDIA/nccl-tests
    algbw = torch.tensor([size / duration]).cuda(local_rank)

    # calculate mean across all ranks
    dist.reduce(algbw, dst=0, op=dist.ReduceOp.SUM)
    algbw /= n

    return algbw

def run(local_rank):

    start_time = time.time()

    is_global_rank_0 = dist.get_rank() == 0
    ranks = dist.get_world_size()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    lower_limit = 30
    upper_limit = 32

    #lower_limit = 15
    #upper_limit = 34
    # 2**15 to 2**34 => 32KB to 16GB
    sizes = [2**x for x in range(lower_limit, upper_limit+1)]

    # this is useful for when one wants to interrupt the run - and still report the best outcome so far
    def sigkill_handler(signum, frame):
         finish()
         sys.exit(1)

    signal.signal(signal.SIGINT, sigkill_handler)

    def finish():
        dist.destroy_process_group()

        if not is_global_rank_0:
            return

        print(f"Device info: {get_device_info()}\n")
        print(f"The average bandwidth of all_reduce over {ranks} ranks ({WARMUPS} warmups / {TRIALS} trials):\n")
        print(f"| payload |    busbw   |    algbw   |")
        print(f"| ------: | ---------: | ---------: |")
        for size in busbw.keys():
            print(f"| {fmt_bytes(size):>7} | {conv_to_GBps(busbw[size]):6.2f}GBps | {conv_to_GBps(algbw[size]):6.2f}GBps |")

        time_delta = time.time() - start_time
        time_str = str(datetime.timedelta(seconds=time_delta)).split(".")[0]
        print(f"Legend: 1KiB = 2**10Bytes, 1MiB = 2**20Bytes, 1GiB = 2**30Bytes")
        print(f"        1GBps = 10**9Bytes per second (networking bw spec convention)")
        print(f"Elapsed time: {time_str}")

    algbw = {}
    busbw = {}
    for size in sizes:
        # clear prev-iteration memory for cards w/ ~24GB
        tensor = None
        gc.collect()

        # /4 is for 4 bytes in fp32
        tensor = torch.rand(size//4, 1, dtype=torch.float32).cuda(local_rank)

        # do a few warm up iterations
        for i in range(WARMUPS):
            timed_allreduce(local_rank,tensor, size, start_event, end_event)

        # real benchmark
        algbw_gather = []
        for i in range(TRIALS):
            if is_global_rank_0:
                print(f"{fmt_bytes(size):>6}: {i+1}", end="\r")
            algbw_gather += timed_allreduce(local_rank,tensor, size, start_event, end_event)
        if is_global_rank_0:
            print()

        algbw[size] = torch.mean(torch.stack(algbw_gather)).item()

        # the 2*(n-1)/n busbw correction factor specific to all-reduce is explained here:
        # https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md#allreduce
        # busbw reflects how optimally the hardware is used
        busbw[size] = algbw[size] * (2*(ranks - 1) / ranks)

    finish()


def init_processes(local_rank, fn, backend='nccl'):
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend)
    fn(local_rank)


if __name__ == "__main__":
    local_rank = int(os.environ["LOCAL_RANK"])
    init_processes(local_rank=local_rank, fn=run)