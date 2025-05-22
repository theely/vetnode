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

Existing vetnode home folders:
- /capstor/scratch/cscs/palmee/vetnode/
- /scratch/ceph/palmee/vetnode


# Slurm Set-up

Add the Alloc flag to PrologFlags (e.g. PrologFlags=Alloc,contain,X11) to enable node vetting at allocation time.

Don't set ReturnToService to 0 or the nodes failing the vetting won't automatically reenter service. 