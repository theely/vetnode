import asyncio
import click
import traceback
import socket  
from shrike.configuration import Configuration

@click.command()
@click.argument("config", type=click.Path())
def diagnose(config) -> None:
    hostname:str = socket.gethostname()
    Configuration._yaml_file = config
    configuration = Configuration()
    click.echo(f"Running sanity checks: {configuration.name} on node:{hostname}")

    results = asyncio.run(run_evals(configuration.evals))
    healthy:bool=True
    click.echo("----------------------------")
    click.echo("Results:")
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

@click.command()
@click.argument("config", type=click.Path())
def setup(config) -> None:
    Configuration._yaml_file = config
    configuration = Configuration()
    
    asyncio.run(run_setups(configuration.evals))


async def run_evals(evals):
    tasks = [eval.eval() for eval in evals]
    return await asyncio.gather(*tasks, return_exceptions=True)


async def run_setups(evals):
    for eval in evals:
        if eval.verify():
            eval.setup()