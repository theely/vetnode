# Vetnode Install
To run vetnode as a Slurm prolog it is advised to pre-install it on the filesystem, this will drastically reduce the execution time.

```
mkdir <vetnode-install folder>
cd <vetnode-install folder>

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install vetnode
```

In the `<vetnode-install folder>` add a `config.yaml` file with the default tests you would like to run as part of the Slurm prolog node vetting.

Next ensure all tests requirements are installed by running

```
vetnode setup <vetnode-install folder>/config.yaml
```