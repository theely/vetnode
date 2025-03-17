from typing import Literal

from shrike.commands.nvidiasmi.models import NvidiaSMIOutput
from shrike.commands.nvidiasmi.nvidia_smi_command import NvidiaSMICommand
from shrike.evaluations.base_eval import BaseEval
from shrike.evaluations.models import Evaluation


class GPUEval(BaseEval):
    name:str
    type: Literal["gpu-eval"]
    max_temp: int
    max_used_memory: float

    async def eval(self)->Evaluation:
        eval:Evaluation = super().eval()
        gpu_info : NvidiaSMIOutput = await NvidiaSMICommand().run()
        eval.passed = len([gpu for gpu in gpu_info.gpus if gpu.temp >= self.max_temp or (gpu.memory_used / gpu.memory_total) > self.max_used_memory ]) == 0
        return eval

