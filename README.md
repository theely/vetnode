# Node Vetting for Distributed Workloads

Ensure allocated nodes are vetted before executing a distributed workload through a series of configurable sanity checks. These checks are designed to detect highly dynamic issues (e.g., GPU temperature) and should be performed immediately before executing the main distributed job.

## Features

- ⚡ **Fast and lightweight**
- 🛠️ **Modular and configurable** 
- 🚀 **Easy to extend**

## Usage

The sanity check should be embedded into your HPC `sbatch` script. It helps differentiate between healthy nodes and those that should be excluded.

```bash
# Run sanity check
srun python -m shrike diagnose ../templates/simple-config.yaml >> sanity-results.txt

# Extract node lists
grep '^Cordon:' sanity-results.txt | awk '{print $2}' > cordoned-nodes.txt
grep '^Vetted:' sanity-results.txt | awk '{print $2}' > vetted-nodes.txt

# Run workload only on vetted (healthy) nodes
srun -N $(wc -l < vetted-nodes.txt) --exclude=./cordoned-nodes.txt hostname
```

## Example: Running a Job

To run a demo job that outputs a list of vetted nodes, follow these steps:

```bash
# SSH into your HPC cluster

# Clone the repository
git clone https://github.com/theely/shrike.git shrike
cd shrike

# Submit the job 
sbatch job-clariden.sh
```

This will execute the sanity check and provide a vetted list of nodes before running the distributed workload.


# Development


## Set-up Python Virtual environement

Create a virtual environment:
```console
python3.11 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run the CLI

```
cd src
python -m shrike diagnose ../templates/simple-config.yaml
```


## Running Tests
From the FirecREST root folder run pytest to execute all unit tests.
```console
source .venv/bin/activate
pip install -r ./requirements.txt -r ./requirements-testing.txt
pytest
```
