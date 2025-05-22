
import asyncio
import os
from typing import Literal


from vetnode.evaluations.base_eval import BaseEval
from ctypes import CDLL
from cuda import cuda, nvrtc
import numpy as np


def _cudaGetErrorEnum(error):
    if isinstance(error, cuda.CUresult):
        err, name = cuda.cuGetErrorName(error)
        return name if err == cuda.CUresult.CUDA_SUCCESS else "<unknown>"
    else:
        raise RuntimeError('Unknown error type: {}'.format(error))
    
saxpy = """\
extern "C" __global__
void saxpy(float a, float *x, float *y, float *out, size_t n)
{
 size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
 if (tid < n) {
   out[tid] = a * x[tid] + y[tid];
 }
}
"""

class CUDAEval(BaseEval):
    name:str
    type: Literal["vetnode.evaluations.cuda_eval.CUDAEval"]
    requirements: Literal[["cuda-python","numpy"]]
    cuda_home: str

    def verify(self)->bool:
        libc = CDLL(f"{self.cuda_home}/libnvrtc.so")
        if libc is None:
            return False
        for filename in os.listdir(self.cuda_home):
            if filename.endswith(".so"):
                lib_path = os.path.join(self.cuda_home, filename)
                CDLL(f"{lib_path}")
        return True


    async def check(self,executor)->tuple[bool,dict]:
        return await asyncio.get_event_loop().run_in_executor(executor, self._check)


    def _check(self)->tuple[bool,dict]:

        (err,) = cuda.cuInit(0)
        self.checkCudaErrors(err)

        err, ndevice = cuda.cuDeviceGetCount()
        self.checkCudaErrors(err)

        err, cuDevice = cuda.cuDeviceGet(0)
        self.checkCudaErrors(err)


        err, major = cuda.cuDeviceGetAttribute(cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cuDevice)
        self.checkCudaErrors(err)
        err, minor = cuda.cuDeviceGetAttribute(cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cuDevice)
        self.checkCudaErrors(err)

        arch_arg = bytes(f'--gpu-architecture=compute_{major}{minor}', 'ascii')

        err, prog =nvrtc.nvrtcCreateProgram(str.encode(saxpy), b"saxpy.cu", 0, [], [])
        self.checkCudaErrors(err)

        opts = [b"--fmad=false", arch_arg]
        (err, )=nvrtc.nvrtcCompileProgram(prog, 2, opts)
        self.checkCudaErrors(err)

        # Get PTX from compilation
        err, ptxSize = nvrtc.nvrtcGetPTXSize(prog)
        self.checkCudaErrors(err)
        ptx = b" " * ptxSize
        (err, )= nvrtc.nvrtcGetPTX(prog, ptx)
        self.checkCudaErrors(err)


        # Create context
        err, context = cuda.cuCtxCreate(0, cuDevice)

        # Load PTX as module data and retrieve function
        ptx = np.char.array(ptx)
        # Note: Incompatible --gpu-architecture would be detected here
        err,module = cuda.cuModuleLoadData(ptx.ctypes.data)
        self.checkCudaErrors(err)
        err, kernel = cuda.cuModuleGetFunction(module, b"saxpy")
        self.checkCudaErrors(err)

        NUM_THREADS = 512  # Threads per block
        NUM_BLOCKS = 32768  # Blocks per grid

        a = np.array([2.0], dtype=np.float32)
        n = np.array(NUM_THREADS * NUM_BLOCKS, dtype=np.uint32)
        bufferSize = n * a.itemsize

        hX = np.random.rand(n).astype(dtype=np.float32)
        hY = np.random.rand(n).astype(dtype=np.float32)
        hOut = np.zeros(n).astype(dtype=np.float32)

        err, dXclass = cuda.cuMemAlloc(bufferSize)
        err, dYclass = cuda.cuMemAlloc(bufferSize)
        err, dOutclass = cuda.cuMemAlloc(bufferSize)

        err, stream = cuda.cuStreamCreate(0)

        cuda.cuMemcpyHtoDAsync(dXclass, hX.ctypes.data, bufferSize, stream)
        cuda.cuMemcpyHtoDAsync(dYclass, hY.ctypes.data, bufferSize, stream)

        dX = np.array([int(dXclass)], dtype=np.uint64)
        dY = np.array([int(dYclass)], dtype=np.uint64)
        dOut = np.array([int(dOutclass)], dtype=np.uint64)

        args = [a, dX, dY, dOut, n]
        args = np.array([arg.ctypes.data for arg in args], dtype=np.uint64)

        (err,) = cuda.cuLaunchKernel(
            kernel,
            NUM_BLOCKS,  # grid x dim
            1,  # grid y dim
            1,  # grid z dim
            NUM_THREADS,  # block x dim
            1,  # block y dim
            1,  # block z dim
            0,  # dynamic shared memory
            stream,  # stream
            args.ctypes.data,  # kernel arguments
            0,  # extra (ignore)
        )
        self.checkCudaErrors(err)

        cuda.cuMemcpyDtoHAsync(hOut.ctypes.data, dOutclass, bufferSize, stream)
        cuda.cuStreamSynchronize(stream)

        hZ = a * hX + hY
        if not np.allclose(hOut, hZ):
            raise ValueError("Error outside tolerance for host-device vectors")
        
        cuda.cuStreamDestroy(stream)
        cuda.cuMemFree(dXclass)
        cuda.cuMemFree(dYclass)
        cuda.cuMemFree(dOutclass)
        cuda.cuModuleUnload(module)
        cuda.cuCtxDestroy(context)

        return True,None

    def checkCudaErrors(self, err):
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError("CUDA error code={}({})".format(err, _cudaGetErrorEnum(err)))
        