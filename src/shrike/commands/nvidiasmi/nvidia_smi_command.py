import asyncio
from importlib import resources as impresources
import textfsm
from shrike.commands import nvidiasmi
from shrike.commands.nvidiasmi.models import NvidiaSMIOutput



class NvidiaSMICommand:
    
   def __init__(self):
      with open(impresources.files(nvidiasmi) / "nvidia-smi-info.tfsm") as template:
         self.parser = textfsm.TextFSM(template)

   
   async def run(self):
       return_code, stdout, stderr = await self._execute()
       return self._parse(stdout)

   def _parse(self, raw_command_output):
      gpus_data = self.parser.ParseText(raw_command_output) #string

      gpus_info = [dict(zip(self.parser.header, row)) for row in gpus_data]
      return NvidiaSMIOutput(**{"gpus":gpus_info})
    
   async def _execute(self): 
      proc = await asyncio.create_subprocess_exec('nvidia-smi', '-q -a',
         stdout=asyncio.subprocess.PIPE,
         stderr=asyncio.subprocess.PIPE)

      stdout, stderr = await proc.communicate()
      await proc.wait()
      return_code = proc.returncode
      return return_code, stdout[0].decode("utf-8"), stderr[0].decode("utf-8")