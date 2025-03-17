import asyncio
import click

from shrike.configuration import Configuration

@click.command()
@click.argument("config", type=click.Path())
def diagnose(config) -> None:

    Configuration._yaml_file = config
    configuration = Configuration()
    click.echo(f"Loading configuration: {configuration.name}")

    results = asyncio.run(run_evals(configuration.evals))

    for result in results:
        if isinstance(result, Exception):
            print(f"Unexpected exception: {result}")
        else:
            click.echo(f"{result}")
        

async def run_evals(evals):
    tasks = [eval.eval() for eval in evals]
    return await asyncio.gather(*tasks, return_exceptions=True)