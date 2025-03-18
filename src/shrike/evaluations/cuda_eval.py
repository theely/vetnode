from typing import Literal

from shrike.evaluations.base_eval import BaseEval
from shrike.evaluations.models import Evaluation

from cuda.bindings import driver, nvrtc


def _cudaGetErrorEnum(error):
    if isinstance(error, driver.CUresult):
        err, name = driver.cuGetErrorName(error)
        return name if err == driver.CUresult.CUDA_SUCCESS else "<unknown>"
    elif isinstance(error, nvrtc.nvrtcResult):
        return nvrtc.nvrtcGetErrorString(error)[1]
    else:
        raise RuntimeError('Unknown error type: {}'.format(error))

class CUDAEval(BaseEval):
    name:str
    type: Literal["cuda-eval"]

    async def eval(self)->Evaluation:
        eval:Evaluation = super().eval()

                # Initialize CUDA Driver API
        self.checkCudaErrors(driver.cuInit(0))

        #self.checkCudaErrors(driver.cuDeviceGetCount())

        # Retrieve handle for device 0
        cuDevice = self.checkCudaErrors(driver.cuDeviceGet(0))

        # Derive target architecture for device 0
        major = self.checkCudaErrors(driver.cuDeviceGetAttribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cuDevice))
        minor = self.checkCudaErrors(driver.cuDeviceGetAttribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cuDevice))
        print(f'--gpu-architecture=compute_{major}{minor}', 'ascii')


        return eval

    def checkCudaErrors(result):
        if result[0].value:
            raise RuntimeError("CUDA error code={}({})".format(result[0].value, _cudaGetErrorEnum(result[0])))
        if len(result) == 1:
            return None
        elif len(result) == 2:
            return result[1]
        else:
            return result[1:]