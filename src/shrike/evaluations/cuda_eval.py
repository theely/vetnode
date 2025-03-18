from typing import Literal

from shrike.evaluations.base_eval import BaseEval
from shrike.evaluations.models import Evaluation

import cuda.cuda as cuda


def _cudaGetErrorEnum(error):
    if isinstance(error, cuda.CUresult):
        err, name = cuda.cuGetErrorName(error)
        return name if err == cuda.CUresult.CUDA_SUCCESS else "<unknown>"
    else:
        raise RuntimeError('Unknown error type: {}'.format(error))

class CUDAEval(BaseEval):
    name:str
    type: Literal["cuda-eval"]

    async def eval(self)->Evaluation:
        eval:Evaluation = super().eval()

        print("Init driver")
        (err,) = cuda.cuInit(0)
        self.checkCudaErrors(err)

        #self.checkCudaErrors(driver.cuDeviceGetCount())


        print("Get device")
        err, cuDevice = cuda.cuDeviceGet(0)
        self.checkCudaErrors(err)

        # Derive target architecture for device 0
        print("get version")    
        err, major = cuda.cuDeviceGetAttribute(cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cuDevice)
        self.checkCudaErrors(err)
        err, minor = cuda.cuDeviceGetAttribute(cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cuDevice)
        self.checkCudaErrors(err)
        print(f'--gpu-architecture=compute_{major}{minor}', 'ascii')


        return eval

    def checkCudaErrors(err):
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError("CUDA error code={}({})".format(err, _cudaGetErrorEnum(err)))
        