


from pydantic import BaseModel


class Evaluation(BaseModel):
   test_name:str
   test_type:str
   passed:bool