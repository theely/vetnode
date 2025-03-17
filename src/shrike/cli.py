import asyncio
import click
import traceback
import socket  
from shrike.configuration import Configuration

@click.command()
@click.argument("config", type=click.Path())
def diagnose(config) -> None:

    Configuration._yaml_file = config
    configuration = Configuration()
    click.echo(f"Loading configuration: {configuration.name}")

    results = asyncio.run(run_evals(configuration.evals))
    failed:bool=False
    for result in results:
        if isinstance(result, Exception):
            #print(f"Unexpected exception: {result}")
            #traceback.print_tb(result.__traceback__)
            failed=True
        else:
            if not result.passed:
                failed=True
            #click.echo(f"{result}")

    if failed:
          click.echo(socket.gethostname())


async def run_evals(evals):
    tasks = [eval.eval() for eval in evals]
    return await asyncio.gather(*tasks, return_exceptions=True)