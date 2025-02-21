
from abc import abstractmethod
from pydantic import BaseModel

from shrike.evaluations.models import Evaluation


class BaseEval(BaseModel):
    name:str
    type:str

    @abstractmethod
    def eval(self)->Evaluation:
        return Evaluation(**{"test_name":self.name, "test_type": self.type, "passed":False})