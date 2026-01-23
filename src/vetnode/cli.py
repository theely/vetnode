import asyncio
from typing import List
import click
import traceback
import socket  
from vetnode.configuration import Configuration
from vetnode.evaluations.models import EvalConfiguration,EvalContext
import os
from vetnode.commands.scontrol.scontrol_command import ScontrolCommand

import subprocess
import sys
from pydoc import locate


def build_context(configuration:Configuration)->EvalContext:
    eval_context:EvalContext    = EvalContext()
    match configuration.scheduler:
            case "slurm":
                eval_context.rank=int(os.environ["SLURM_PROCID"])
                eval_context.local_rank = int(os.environ["SLURM_LOCALID"])
                eval_context.nodes = asyncio.run(ScontrolCommand().run()).hostnames
                eval_context.master_addr = eval_context.nodes[0]
                eval_context.master_port = 29500 #Default port used to collect evaluation results
                eval_context.world_size = int(os.environ['SLURM_NTASKS'])
                eval_context.nodes_count = int(os.environ['SLURM_JOB_NUM_NODES'])
                eval_context.tasks_per_node = int(eval_context.world_size/eval_context.nodes_count)
            case _:
                raise NotImplementedError("Support for the rquested scheduler has not been implemented.")
    return eval_context

@click.command()
@click.argument("config", type=click.Path(exists=True))
def diagnose(config) -> None:
    hostname:str = socket.gethostname()
    Configuration._yaml_file = config
    configuration = Configuration()
    eval_context= build_context(configuration)
    
    click.echo(f"Running sanity checks: {configuration.name} on node:{hostname}")
    evals = load_evals(eval_context, configuration.evals)
    results = asyncio.run(run_evals(evals))
    healthy:bool=True
    click.echo("----------------------------")
    click.echo("** Results:               **")
    click.echo("----------------------------")
    for result in results:
        if isinstance(result, Exception):
            click.secho(f"Node: {hostname} \t unexpected exception: {result}", fg='red')
            traceback.print_tb(result.__traceback__)
            healthy=False
        else:
            if result.passed is None:
                continue
            if not result.passed:
                healthy=False
            click.secho(f"Node: {hostname} \t result:{result}", fg='green' if result.passed else 'red')

    if healthy:
        click.echo(f"Vetted: {hostname}")
    else:
        click.echo(f"Cordon: {hostname}")
        sys.exit(1)

@click.command()
@click.argument("config", type=click.Path(exists=True))
def setup(config) -> None:
    Configuration._yaml_file = config
    configuration = Configuration()
    eval_context= build_context(configuration)
    click.echo("----------------------------")
    click.echo("** Tests initialization!  **")
    click.echo("----------------------------")
    load_evals(eval_context,configuration.evals, install=True, index_url=configuration.pip.index_url)


async def run_evals(evals,):
    tasks = []
    for eval in evals:
        try:
            if eval.verify():
                tasks.append(eval.eval())
        except Exception as ex:
            click.secho(f"Skipped: {eval.name} (error: {ex})", fg='red')
    return await asyncio.gather(*tasks, return_exceptions=True)


def load_evals( base_eval_context:EvalContext, eval_configs: List[EvalConfiguration], install:bool=False,index_url: str = None):
    evals = []
    for i,eval in enumerate(eval_configs):
        eval_context = base_eval_context.model_copy(update={'eval_id': i})
        #Load class dynamically
        try:
            if install and eval.requirements:
                load_requirements(eval.requirements,index_url)
            eval_class = locate(eval.type)
            evals.append(eval_class(eval_context,**eval.model_dump()))
        except Exception as ex:
            click.secho(f"Skipped: {eval.name} (error: {ex})", fg='red')
    return evals

def load_requirements(requirements: List[str], index_url: str = None):
    for package in requirements:
        cmd = [sys.executable, "-m", "pip", "install", "-q"]
        if index_url:
            cmd += ["--index-url",index_url]
        if isinstance(package, str):
            cmd.append(package)
        else:
            cmd += package
        subprocess.check_call(cmd)