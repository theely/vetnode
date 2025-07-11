import asyncio
from typing import List
import click
import traceback
import socket  
from vetnode.configuration import Configuration
from vetnode.evaluations.models import EvalConfiguration

import subprocess
import sys
from pydoc import locate

@click.command()
@click.argument("config", type=click.Path(exists=True))
def diagnose(config) -> None:
    hostname:str = socket.gethostname()
    Configuration._yaml_file = config
    configuration = Configuration()
    click.echo(f"Running sanity checks: {configuration.name} on node:{hostname}")
    evals = load_evals(configuration.evals)
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
    click.echo("----------------------------")
    click.echo("** Tests initialization!  **")
    click.echo("----------------------------")
    load_evals(configuration.evals, install=True, index_url=configuration.pip.index_url)


async def run_evals(evals):
    tasks = []
    for eval in evals:
        try:
            if eval.verify():
                tasks.append(eval.eval())
        except Exception as ex:
            click.secho(f"Skipped: {eval.name} (error: {ex})", fg='red')
    return await asyncio.gather(*tasks, return_exceptions=True)


def load_evals(eval_configs: List[EvalConfiguration], install:bool=False,index_url: str = None):
    evals = []
    for eval in eval_configs:
        
        #Load class dynamically
        try:
            if install and eval.requirements:
                load_requirements(eval.requirements,index_url)
            eval_class = locate(eval.type)
            evals.append(eval_class(**eval.model_dump()))
        except Exception as ex:
            click.secho(f"Skipped: {eval.name} (error: {ex})", fg='red')
    return evals

def load_requirements(requirements: List[str], index_url: str = None):
    for package in requirements:
        cmd = [sys.executable, "-m", "pip", "install"]
        if index_url:
            cmd += ["--index-url",index_url]
        if isinstance(package, str):
            cmd.append(package)
        else:
            cmd += package
        subprocess.check_call(cmd)