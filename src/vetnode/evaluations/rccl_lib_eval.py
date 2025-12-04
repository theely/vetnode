
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
from vetnode.evaluations.models import BandwidthSize, BinaryByteSize
import numpy as np
import traceback
from hip import hip, rccl
import ctypes

# Define NCCL constants
ncclUniqueId_t = ctypes.c_byte * 128
ncclComm_t = ctypes.c_void_p
cudaStream_t = ctypes.c_void_p

conv_to_GBps = lambda v : v/10**9

class RCCLEvalWarmUp(BaseModel):
    payload:BinaryByteSize= '256 MB'
    runs:int= 3

class RcclLibEval(BaseEval):
    name:str
    type: Literal["vetnode.evaluations.rccl_lib_eval.RcclLibEval"]
    requirements: Literal[[['hip-python~=7.0.2','--index-url','https://test.pypi.org/simple'] ,"numpy"]]
    scheduler:  Literal["slurm"]
    payload: BinaryByteSize = '4 GB'
    method: Literal["allreduce"] = "allreduce"
    warmup: RCCLEvalWarmUp
    min_bandwidth: BandwidthSize = '15 GB/s'
    
    def verify(self)->bool:
        libs =["/opt/rocm/lib/librccl.so"]   #add lib librccl-net.so "libnvrtc.so"
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

    def hip_check(self, call_result):
        err = call_result[0]
        result = call_result[1:]
        if len(result) == 1:
            result = result[0]
        if isinstance(err, hip.hipError_t) and err != hip.hipError_t.hipSuccess:
            raise RuntimeError(str(err))
        if isinstance(err, rccl.ncclResult_t) and err != rccl.ncclResult_t.ncclSuccess:
            raise RuntimeError(str(err))
        return result


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

        nccl = ctypes.cdll.LoadLibrary('/opt/rocm/lib/librccl.so')
        

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
        
        uid = rccl.ncclUniqueId()
        if rank==0 and local_rank==0:
            print(f"Get uniqueid: {rank}")
            rccl.ncclGetUniqueId(uid)     
            print(f"Broadcast uniqueid")       
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('0.0.0.0', 13333))
                s.settimeout(30) #wait 30s for clients to connect
                s.listen()
                for _ in range(world_size-1):
                    conn, _ = s.accept()
                    with conn:
                        conn.send(rccl.ncclUniqueId.fromPyobj(uid).getElementPtr())
        else:
            for i in range(5):
                try:
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                        s.connect((master_node, 13333))
                        s.recv_into(rccl.ncclUniqueId.fromPyobj(uid).getElementPtr())
                        break
                except socket.error:
                    time.sleep(1)
                
    

        # Get current device
        device=self.hip_check( hip.hipGetDevice())

        self.hip_check(hip.hipSetDevice(np.int32(local_rank)))
        stream = self.hip_check(hip.hipStreamCreate())

        stream_ptr = ctypes.c_void_p(int(stream))


        print(f"init rank: {rank}")
        result, comm = rccl.ncclCommInitRank(world_size, uid, int(rank))
        print(f"Comm initialized: {comm}")

        
        #result = nccl.ncclCommInitRank(ctypes.byref(comm), world_size, uid, int(rank))
        #if result != 0:
        #    error_str = nccl.ncclGetErrorString(result)
        #    return False, {"error": f"NCCL error: {error_str.decode('utf-8')}"}
        
        # Warm-up phase
        n = self.warmup.payload//4 #np.float32 is 4 baytes
        host = np.full(n, rank + 1, dtype=np.float32)
        status, dev_in = hip.hipMalloc(host.nbytes)
        status, dev_out = hip.hipMalloc(host.nbytes)
        hip.hipMemcpy(dev_in,host.ctypes.data,host.nbytes,hip.hipMemcpyKind.hipMemcpyHostToDevice)
        for _ in range(self.warmup.runs):
            rccl.ncclAllReduce(dev_in, dev_out, n, rccl.ncclDataType_t.ncclFloat32, rccl.ncclRedOp_t.ncclSum, comm, stream)

        # Actual measurement
        n = self.payload//4 #np.float32 is 4 baytes
        
        host = np.full(n, rank + 1, dtype=np.float32)
        status, dev_in = hip.cudaMalloc(host.nbytes)
        status, dev_out = hip.cudaMalloc(host.nbytes)
        hip.cudaMemcpy(dev_in, host.ctypes.data, host.nbytes, hip.hipMemcpyKind.hipMemcpyDeviceToHost)

        start_time = time.time()

        result = rccl.ncclAllReduce(dev_in, dev_out, n, rccl.ncclDataType_t.ncclFloat32, rccl.ncclRedOp_t.ncclSum, comm, stream)
        if result != 0:
            error_str = nccl.ncclGetErrorString(result)
            return False, {"error": f"NCCL error: {error_str.decode('utf-8')}"}
        
        hip.cudaStreamSynchronize(stream)
        end_time = time.time()
        elapsedtime = end_time-start_time
   
        hip.hipFree(dev_in)
        hip.hipFree(dev_out)

        nccl.ncclCommDestroy(comm)
        bandwidth = (self.payload/elapsedtime) * (2*(world_size - 1) / world_size)   
        return bandwidth > self.min_bandwidth, {"bandwidth": f"{conv_to_GBps(bandwidth):6.2f} GB/s"}
