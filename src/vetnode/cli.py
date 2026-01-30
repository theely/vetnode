import asyncio
from typing import List
import click
import traceback
import socket  
from vetnode.configuration import Configuration
from vetnode.evaluations.models import EvalConfiguration,EvalContext,EvalResult
import os
from vetnode.commands.scontrol.scontrol_command import ScontrolCommand
import asyncio
import struct
import sys
import subprocess
import sys
from pydoc import locate



def build_context(configuration:Configuration)->EvalContext:
    eval_context:EvalContext    = EvalContext()
    eval_context.scheduler = configuration.scheduler
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
            case "standalone":
                eval_context.rank=None
                eval_context.local_rank = None
                eval_context.nodes = None
                eval_context.master_addr = "localhost"
                eval_context.master_port = 29500
                eval_context.world_size = 1
                eval_context.nodes_count = None
                eval_context.tasks_per_node = None
            case _:
                raise NotImplementedError("Support for the rquested scheduler has not been implemented.")
    return eval_context

@click.command()
@click.argument("config", type=click.Path(exists=True))
def diagnose(config) -> None:
    hostname:str = socket.gethostname()
    Configuration._yaml_file = config
    configuration = Configuration()
    main_context= build_context(configuration)
    
    click.echo(f"Running sanity checks: {configuration.name} on node:{hostname}")
    evals = load_evals(main_context, configuration.evals)
    processes = asyncio.run(run_evals(main_context,evals))
    healthy:bool=True
    for results in processes:
        if isinstance(results, List):
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
    load_evals(eval_context,configuration.evals, install=True, index_url=configuration.pip.index_url)


async def send_int(writer, value: int):
    writer.write(struct.pack("!i", value))
    await writer.drain()

async def recv_int(reader) -> int:
    data = await reader.readexactly(4)
    return struct.unpack("!i", data)[0]

async def send_str(writer, msg: str):
    data = msg.encode()
    await send_int(writer, len(data))
    writer.write(data)
    await writer.drain()

async def recv_str(reader) -> str:
    length = await recv_int(reader)
    data = await reader.readexactly(length)
    return data.decode()


async def run_evals(main_context,evals):
    tasks = []
    if main_context.rank==0 and main_context.local_rank==0:
        tasks.append(asyncio.create_task(synchronize_workers(main_context,evals)))
    
    tasks.append(asyncio.create_task(run_evals_worker(main_context,evals)))
    return await asyncio.gather(*tasks, return_exceptions=True)





async def run_evals_worker(main_context,evals):
    results = []
    for attempt in range(10):
        try:
            reader, writer = await asyncio.wait_for(asyncio.open_connection(main_context.master_addr, main_context.master_port), timeout=5.0)
            try:
                while True:
                    i = await recv_int(reader)
                    print(f"recived int: {i}")
                    if i == -1:
                        return results

                    await asyncio.sleep(1.0)
                    
                    eval = evals[i]
                    click.secho(f"Starting evaluation {eval.name}", fg='blue')
                    result = None
                    try:
                        if eval.verify():
                            result = await eval.eval()
                            results.append(result)
                    except Exception as ex:
                        click.secho(f"Skipped: {eval.name} (error: {ex})", fg='red')
                    str = result.model_dump_json()
                    click.secho(f"Sending result: {str}", fg='red')
                    await send_str(writer, f"{str}\n")
            finally:
                writer.close()
                await writer.wait_closed()
                return results
        except (asyncio.TimeoutError, ConnectionRefusedError) as e:
            await asyncio.sleep(2.0)
        except Exception as e:
            click.secho(f"Worker encountered an unexpected error: {e}", fg='red')
            traceback.print_exc()
            await asyncio.sleep(2.0)
    raise ConnectionError(
        f"Failed to connect to master node {eval_context.master_addr}:{eval_context.master_port} after {retries} attempts"
    )


async def synchronize_workers(main_context,evals):
    clients = []  # [(reader, writer)]
    results = [[None] * main_context.world_size  for _ in range(len(evals))]  # Initialize results list with None values
    run = True
    print(f"results init: {results}")
    async def handle_client(reader, writer):
        clients.append((reader, writer))
        click.secho(f"Worker ready ({len(clients)}/{main_context.world_size})", fg='green')
        while run:
            try:
                result_json = await recv_str(reader)
                click.secho(f"Received result from worker: {result_json}", fg='white')
                result = EvalResult.model_validate_json(result_json)
                results[result.eval_id][result.rank] = result
            except Exception as e:
                click.secho(f"Worker disconnected. Error: {e}", fg='red')
                break


    server = await asyncio.start_server(handle_client, '0.0.0.0', main_context.master_port)
    click.secho(f"Waiting for {main_context.world_size} workers", fg='white')
    async with server:
        while len(clients) < main_context.world_size:
            await asyncio.sleep(0.2)

        for i in range(len(evals)):

            # Send task index
            click.secho(f"\nEvaluating {evals[i].name}:", fg='blue')
            for _, writer in clients:
                await send_int(writer, i)

            # Wait for all clients
            while run:
                await asyncio.sleep(5.0)
                print(f"results: {results}")
                if sum(result is not None for result in results[i]) >= main_context.world_size:
                    break
                

        # Shutdown
        print("Shutting down master")
        run = False
        for _, writer in clients:
            await send_int(writer, -1)
        await asyncio.sleep(5.0)
        for reader, writer in clients:
            reader.close()
            writer.close()
            await writer.wait_closed()



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