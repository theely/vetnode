# Add src folder to python paths
from importlib import resources as impresources

import textfsm
from tests import mocked_commands
from src.vetnode.commands import nvidiasmi, scontrol
import pytest

def load_cmd_output(file: str):
    return impresources.files(mocked_commands) / file
    
def load_template(file: str):
    return impresources.files(nvidiasmi) / file
    

nvidia_smi_logs = [load_cmd_output("nvidia-smi-log-GH200.txt"),load_cmd_output("nvidia-smi-log-A100.txt")]

@pytest.fixture(scope="module")
def nvidia_smi_template():
    return impresources.files(nvidiasmi) /"nvidia-smi-info.tfsm"

@pytest.fixture(scope="module")
def scontrol_log():
    return load_cmd_output("scontrol-log.txt")

@pytest.fixture(scope="module")
def scontrol_template():
    return impresources.files(scontrol) /"scontrol-hostnames.tfsm"

@pytest.mark.parametrize('nvidia_smi_log', nvidia_smi_logs)
def test_nvidia_smi_log(nvidia_smi_log,nvidia_smi_template):
   
    with open(nvidia_smi_template) as template, open(nvidia_smi_log) as output:
        re_table = textfsm.TextFSM(template)
        data = re_table.ParseText(output.read())

        assert (len(data)== 4)
        assert (data[0][0]== '00000009:01:00.0')
        assert (data[0][1]== '1')
        assert (data[0][2]== 'NVIDIA GH200 120GB' or data[0][2]== 'NVIDIA A100-SXM4-80GB')
        assert (data[0][3]== '23')
        assert (data[0][4]== '97871')
        assert (data[0][5]== '284')


def test_scontrol_log(scontrol_log,scontrol_template):
   
    with open(scontrol_template) as template, open(scontrol_log) as output:
        re_table = textfsm.TextFSM(template)
        data = re_table.ParseText(output.read())

        assert (len(data)== 5)
        assert (data[0][0]== 'localhost1')
        assert (data[1][0]== 'localhost2')
        assert (data[2][0]== 'localhost3')
        assert (data[3][0]== 'localhost4')
        assert (data[4][0]== 'localhost5')

    
