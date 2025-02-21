import asyncio
import click

from shrike.configuration import Configuration

@click.command()
@click.argument("config", type=click.Path())
def diagnose(config) -> None:

    Configuration._yaml_file = config
    configuration = Configuration()
    click.echo(f"Hello world: {configuration.name}")

    results = asyncio.run(run_evals(configuration.evals))

    #click.secho('Hello World!', fg='green')
    #click.secho('Some more text', bg='blue', fg='white')
    #click.secho('ATTENTION', blink=True, bold=True)

    for result in results:
        click.echo(f"{result}")
        

async def run_evals(evals):
    tasks = [eval.eval() for eval in evals]
    return await asyncio.gather(*tasks, return_exceptions=True)