from typing import Literal, Optional

import click
import socket
from shrike.evaluations.base_eval import BaseEval
from shrike.evaluations.models import Evaluation

class InternetAccessEval(BaseEval):
    name:str
    type: Literal["internet-access"]
    host:str
    port:Optional[int]=53
    timeout:Optional[int]=3

    async def check(self)->Evaluation:
        eval:Evaluation = super().eval()
        try:
            socket.setdefaulttimeout(self.timeout)
            socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((self.host, self.port))
            eval.passed = True
            return eval
        except socket.error as ex:
            click.echo(f"Exception: {ex}")
            return eval