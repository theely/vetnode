
import asyncio
import base64
import os
import time
from typing import Literal
import click
from pydantic import BaseModel
import ctypes, socket
from vetnode.commands.scontrol.scontrol_command import ScontrolCommand
from vetnode.evaluations.base_eval import BaseEval
from vetnode.evaluations.models import BandwithSize, BinaryByteSize
import numpy as np
import traceback
import cuda.bindings.runtime as cudart

# Define NCCL constants
ncclUniqueId_t = ctypes.c_byte * 128
ncclComm_t = ctypes.c_void_p
cudaStream_t = ctypes.c_void_p

conv_to_GBps = lambda v : v/10**9

class NCCLEvalWarmUp(BaseModel):
    payload:BinaryByteSize= '256 MB'
    runs:int= 3

class NcclLibEval(BaseEval):
    name:str
    type: Literal["vetnode.evaluations.nccl_lib_eval.NcclLibEval"]
    requirements: Literal[["cuda-python","numpy"]]
    scheduler:  Literal["slurm"]
    payload: BinaryByteSize = '4 GB'
    method: Literal["broadcast"] = "broadcast"
    warmup: NCCLEvalWarmUp
    min_bandwidth: BandwithSize = '15 GB/s'
    
    def verify(self)->bool:
        libs =["libnvrtc.so","libnccl.so"]   #add lib libnccl-net.so
        for lib in libs:
            libc = ctypes.CDLL(lib)
            if libc is None:
                return False
        return True

    async def check(self,executor)->bool:
        try:
            return await asyncio.get_event_loop().run_in_executor(executor, self._check)
        except Exception as e:
            click.echo(f"Error executing check: {e}")
            traceback.print_exc()


    def _check(self)->tuple[bool,dict]:
        local_rank =None
        rank= None
        nodes = None
        master_node = None
        world_size =None
        match self.scheduler:
            case "slurm":
                rank = int(os.environ["SLURM_PROCID"])
                local_rank = int(os.environ["SLURM_LOCALID"])
                nodes = asyncio.run(ScontrolCommand().run()).hostnames
                master_node = nodes[0]
                world_size = int(os.environ['SLURM_NTASKS'])
            case _:
                raise NotImplementedError("Support for the rquested scheduler has not been implemented.")

        nccl = ctypes.cdll.LoadLibrary('libnccl.so')
        

        #TODO: re-implement following: https://github.com/vllm-project/vllm/blob/main/vllm/distributed/device_communicators/pynccl_wrapper.py#L49

        # Define API prototypes
        nccl.ncclGetUniqueId.restype = ctypes.c_int
        nccl.ncclGetUniqueId.argtypes = [ctypes.POINTER(ncclUniqueId_t)]

        nccl.ncclCommInitRank.restype = ctypes.c_int
        nccl.ncclCommInitRank.argtypes = [ctypes.POINTER(ncclComm_t), ctypes.c_int, ncclUniqueId_t, ctypes.c_int]

        nccl.ncclAllReduce.restype = ctypes.c_int
        nccl.ncclAllReduce.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
                                    ctypes.c_int, ctypes.c_int, ncclComm_t, cudaStream_t]
        
        nccl.ncclGetErrorString.restype = ctypes.c_char_p
        nccl.ncclGetErrorString.argtypes = [ctypes.c_int]
        
        nccl.ncclBroadcast.restype = ctypes.c_int
        nccl.ncclBroadcast.argtypes = [
            ctypes.c_void_p,  # sendbuf
            ctypes.c_void_p,  # recvbuf
            ctypes.c_size_t,  # count
            ctypes.c_int,     # datatype
            ctypes.c_int,     # root
            ncclComm_t,       # comm
            cudaStream_t,  # stream
        ]

        nccl.ncclCommDestroy.restype = ctypes.c_int
        nccl.ncclCommDestroy.argtypes = [ncclComm_t]
                
        ncclDataType_t = 7  # ncclFloat32
        ncclRedOp_t = 0     # ncclSum
        
        uid = ncclUniqueId_t()
        if rank==0:
            nccl.ncclGetUniqueId(ctypes.byref(uid))            
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('0.0.0.0', 13333))
                s.settimeout(30) #wait 30s for clients to connect
                s.listen()
                for _ in range(world_size-1):
                    conn, _ = s.accept()
                    with conn:
                        conn.send(uid)
        else:
            for i in range(5):
                try:
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                        s.connect((master_node, 13333))
                        s.recv_into(uid)
                        break
                except socket.error:
                    time.sleep(1)
                
        cudart.cudaGetDevice()
        (err,) = cudart.cudaSetDevice(0)
        assert err == 0

        err, stream = cudart.cudaStreamCreate()
        assert err == 0
        stream_ptr = ctypes.c_void_p(int(stream))


        comm = ncclComm_t()
        result = nccl.ncclCommInitRank(ctypes.byref(comm), world_size, uid, rank)
        if result != 0:
            error_str = nccl.ncclGetErrorString(result)
            return False, {"error": f"NCCL error: {error_str.decode('utf-8')}"}
        
        n = self.payload//4 #np.float32 is 4 baytes
        
        host = np.full(n, rank + 1, dtype=np.float32)
        status, dev_in = cudart.cudaMalloc(host.nbytes)
        status, dev_out = cudart.cudaMalloc(host.nbytes)
        cudart.cudaMemcpy(dev_in, host.ctypes.data, host.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

        start_time = time.time()
        result = nccl.ncclAllReduce(dev_in, dev_out, n, ncclDataType_t, ncclRedOp_t, comm, stream_ptr)
        if result != 0:
            error_str = nccl.ncclGetErrorString(result)
            return False, {"error": f"NCCL error: {error_str.decode('utf-8')}"}
        
        cudart.cudaStreamSynchronize(stream)
        end_time = time.time()
        elapsedtime = end_time-start_time
   
        nccl.ncclCommDestroy(comm)
        bandwith = self.payload/elapsedtime   
        return bandwith > self.min_bandwidth, {"Bandwidth": f"{conv_to_GBps(bandwith):6.2f} GB/s"}

