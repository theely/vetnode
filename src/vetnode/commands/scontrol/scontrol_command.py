import asyncio
from importlib import resources as impresources
import textfsm
from vetnode.commands import scontrol
from vetnode.commands.scontrol.models import ScontrolOutput



class ScontrolCommand:
    
   def __init__(self):
      with open(impresources.files(scontrol) / "scontrol-hostnames.tfsm") as template:
         self.parser = textfsm.TextFSM(template)

   
   async def run(self):
       return_code, stdout, stderr = await self._execute()
       if return_code != 0:
          raise ValueError("Scontrol command return code is non zero.")
       return self._parse(stdout)

   def _parse(self, raw_command_output):
      hostnames = self.parser.ParseText(raw_command_output) #string
      return ScontrolOutput(**{"hostnames":[row[0] for row in hostnames]})
    
   async def _execute(self): 
      cmd = ['scontrol', 'show', 'hostnames', '$SLURM_JOB_NODELIST']
      proc = await asyncio.create_subprocess_shell(' '.join(cmd),
         stdout=asyncio.subprocess.PIPE,
         stderr=asyncio.subprocess.PIPE)

      stdout, stderr = await proc.communicate()
      await proc.wait()
      return_code = proc.returncode
      return return_code, stdout.decode(), stderr.decode()