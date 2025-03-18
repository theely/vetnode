from typing import Literal

from shrike.evaluations.base_eval import BaseEval
from shrike.evaluations.models import Evaluation

from cuda import cuda, nvrtc



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
    type: Literal["cuda-eval"]

    async def eval(self)->Evaluation:
        eval:Evaluation = super().eval()


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

        eval.passed = True
        return eval

    def checkCudaErrors(self, err):
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError("CUDA error code={}({})".format(err, _cudaGetErrorEnum(err)))
        