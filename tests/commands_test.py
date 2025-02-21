# Add src folder to python paths
from importlib import resources as impresources

import textfsm
from tests import mocked_commands
from src.shrike.commands import nvidiasmi
import pytest

def load_cmd_output(file: str):
    return impresources.files(mocked_commands) / file
    
def load_template(file: str):
    return impresources.files(nvidiasmi) / file
    

@pytest.fixture(scope="module")
def nvidia_smi_log():
    return load_cmd_output("nvidia-smi-log.txt")

@pytest.fixture(scope="module")
def nvidia_smi_template():
    return load_template("nvidia-smi-info.tfsm")


def test_nvidia_smi_log(nvidia_smi_log,nvidia_smi_template):
   
    with open(nvidia_smi_template) as template, open(nvidia_smi_log) as output:
        re_table = textfsm.TextFSM(template)
        data = re_table.ParseText(output.read())

        assert (len(data)== 4)
        assert (data[0][0]== '00000009:01:00.0')
        assert (data[0][1]== '1')
        assert (data[0][2]== 'NVIDIA GH200 120GB')
        assert (data[0][3]== '23')
        assert (data[0][4]== '97871')
        assert (data[0][5]== '284')


    
