# Node Vetting for Distributed Workloads

Ensure allocated nodes are vetted before executing a distributed workload through a series of configurable sanity checks. These checks are designed to detect highly dynamic issues (e.g., GPU temperature) and should be performed immediately before executing the main distributed job.

## Features

- ⚡ **Fast and lightweight**
- 🛠️ **Modular and configurable** 
- 🚀 **Easy to extend**

## Getting Started

```bash
# Install
pip install vetnode

# checks for dependencies and installs requirements
vetnode setup ./examples/local-test/config.yaml

# runs the vetting process
vetnode diagnose ./examples/local-test/config.yaml
```

## Workflow Usage Example

The vetnode cli is intended to be embedded into your HPC workflow. 
The following is a node vetting example for a ML (machine learning) workflow on a Slurm HPC cluster.

```bash

#!/bin/bash

#SBATCH --nodes=6
#SBATCH --time=0-00:15:00
#SBATCH --account=a-csstaff

REQUIRED_NODES=4
MAIN_JOB_COMMAND="python -m torch.distributed.torchrun --nproc_per_node=$(wc -l < vetted-nodes.txt) main.py"

vetnode setup ../examples/slurm-ml-vetting/config.yaml
srun vetnode diagnose ../examples/slurm-ml-vetting/config.yaml >> results.txt

# Extract node lists
grep '^Cordon:' results.txt | awk '{print $2}' > cordoned-nodes.txt
grep '^Vetted:' results.txt | awk '{print $2}' > vetted-nodes.txt

#Run on healthy nodes only
if [ $(wc -l < vetted-nodes.txt) -ge $REQUIRED_NODES ]; then
    srun -N $REQUIRED_NODES --exclude=./cordoned-nodes.txt $MAIN_JOB_COMMAND
else
    echo "Job canceled!"
    echo "Reason: too few vetted nodes."
fi
```
### Quick Run

The following is a Slurm job example you can download and run as a test.

```bash
curl -o job.sh  https://raw.githubusercontent.com/theely/vetnode/refs/heads/main/examples/slurm-ml-vetting/job.sh
sbatch --account=a-csstaff job.sh

#check job status
squeue -j {jobid} --long

#check vetting results
cat vetnode-{jobid}/results.txt
```

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
python -m vetnode setup ../examples/local-test/config.yaml
python -m vetnode diagnose ../examples/local-test/config.yaml
```


## Running Tests
From the FirecREST root folder run pytest to execute all unit tests.
```console
source .venv/bin/activate
pip install -r ./requirements.txt -r ./requirements-testing.txt
pytest
```

## Distribute

Update version in pyproject.toml file.

```
pip install -r ./requirements-testing.txt
python3 -m build --wheel
twine upload dist/* 
```
Note: API token is sotred in local file .pypirc




# Info dump

Clariden Distro:

NAME="SLES"
VERSION="15-SP5"
VERSION_ID="15.5"
PRETTY_NAME="SUSE Linux Enterprise Server 15 SP5"
ID="sles"
ID_LIKE="suse"
ANSI_COLOR="0;32"
CPE_NAME="cpe:/o:suse:sles:15:sp5"
DOCUMENTATION_URL="https://documentation.suse.com/"


./configure --prefix=/users/palmee/aws-ofi-nccl/install --disable-tests --without-mpi --enable-cudart-dynamic --with-libfabric=/opt/cray/libfabric/1.15.2.0 --with-cuda=/opt/nvidia/hpc_sdk/Linux_aarch64/24.3/cuda/12.3/    



https://download.opensuse.org/repositories/home:/aeszter/openSUSE_Leap_15.3/x86_64/libhwloc5-1.11.8-lp153.1.1.x86_64.rpm
https://download.opensuse.org/repositories/home:/aeszter/15.5/x86_64/libhwloc5-1.11.8-lp155.1.1.x86_64.rpm


## Build plugin in image
export DOCKER_DEFAULT_PLATFORM=linux/amd64 export 
#export DOCKER_DEFAULT_PLATFORM=linux/arm64
docker run -i -t registry.suse.com/suse/sle15:15.5      
zypper install -y libtool git gcc awk make wget

zypper addrepo https://developer.download.nvidia.com/compute/cuda/repos/opensuse15/x86_64/cuda-opensuse15.repo
zypper addrepo https://developer.download.nvidia.com/compute/cuda/repos/sles15/sbsa/cuda-sles15.repo
zypper --non-interactive --gpg-auto-import-keys refresh
zypper install -y cuda-toolkit-12-3

## Add missing lib path required by hwloc
echo "/usr/local/cuda/targets/x86_64-linux/lib/stubs/" | tee /etc/ld.so.conf.d/nvidiaml-x86_64.conf
echo "/usr/local/cuda/targets/sbsa-linux/lib/stubs/" | tee /etc/ld.so.conf.d/nvidiaml-sbsa.conf

ldconfig
ldconfig -p | grep libnvidia

git clone -b v1.19.0 https://github.com/ofiwg/libfabric.git
cd libfabric
autoupdate
./autogen.sh
#CC=gcc ./configure --prefix=/users/palmee/libfabric/install
CC=gcc ./configure
make
make install

wget https://download.open-mpi.org/release/hwloc/v2.12/hwloc-2.12.0.tar.gz
tar -xvzf hwloc-2.12.0.tar.gz 
cd hwloc-2.12.0
#CC=gcc ./configure --prefix=/users/palmee/hwloc-2.12.0/install
CC=gcc ./configure
make 
make install


git clone -b v1.14.0 https://github.com/aws/aws-ofi-nccl.git
cd aws-ofi-nccl
mkdir install
GIT_COMMIT=$(git rev-parse --short HEAD)

./autogen.sh
CC=gcc ./configure --disable-tests --without-mpi \
          --enable-cudart-dynamic  \
          --prefix=./install/v1.14.0-${GIT_COMMIT}/x86_64/12.3/ \
          --with-cuda=/usr/local/cuda 



TODO:
consider building an rpm: https://www.redhat.com/en/blog/create-rpm-package



CC=gcc ./configure --disable-tests --without-mpi \
          --enable-cudart-dynamic  \
          --prefix=/users/palmee/aws-ofi-nccl/install_2/ \
          --with-libfabric=/opt/cray/libfabric/1.15.2.0 --with-cuda=/opt/nvidia/hpc_sdk/Linux_aarch64/24.3/cuda/12.3/ --with-hwloc=/users/palmee/hwloc-2.12.0/install



export LD_LIBRARY_PATH=/opt/cray/libfabric/1.15.2.0/lib64/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_aarch64/24.3/cuda/12.3/lib64/:$LD_LIBRARY_PATH
ld /users/palmee/aws-ofi-nccl/install_2/lib/libnccl-net.so 


## Install NCCL

git clone https://github.com/NVIDIA/nccl.git
git checkout v2.20.3-1  #looks like this is the version compatible with cuda/12.3/
cd nccl
make src.build CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_aarch64/24.3/cuda/12.3/



## WORKING LIB (job 327119)

| payload |    busbw   |    algbw   |
| ------: | ---------: | ---------: |
|    1GiB |  90.94GBps |  46.94GBps |
|    2GiB |  91.24GBps |  47.09GBps |
|    4GiB |  91.35GBps |  47.15GBps |

wget https://download.open-mpi.org/release/hwloc/v2.12/hwloc-2.12.0.tar.gz
tar -xvzf hwloc-2.12.0.tar.gz 
cd hwloc-2.12.0
./configure --prefix=/users/palmee/hwloc-2.12.0/install
make 
make install

git clone -b v1.14.0 https://github.com/aws/aws-ofi-nccl.git
cd aws-ofi-nccl
mkdir install

./autogen.sh
./configure --disable-tests --without-mpi \
          --enable-cudart-dynamic  \
          --prefix=/users/palmee/aws-ofi-nccl/install/ \
          --with-libfabric=/opt/cray/libfabric/1.15.2.0 --with-cuda=/opt/nvidia/hpc_sdk/Linux_aarch64/24.3/cuda/12.3/ --with-hwloc=/users/palmee/hwloc-2.12.0/install


## TEST with gcc (job 327124) - working

| payload |    busbw   |    algbw   |
| ------: | ---------: | ---------: |
|    1GiB |  91.06GBps |  47.00GBps |
|    2GiB |  91.24GBps |  47.09GBps |
|    4GiB |  91.34GBps |  47.15GBps |

wget https://download.open-mpi.org/release/hwloc/v2.12/hwloc-2.12.0.tar.gz
tar -xvzf hwloc-2.12.0.tar.gz 
cd hwloc-2.12.0
CC=gcc  ./configure --prefix=/users/palmee/hwloc-2.12.0/install
make 
make install

git clone -b v1.14.0 https://github.com/aws/aws-ofi-nccl.git
cd aws-ofi-nccl
mkdir install

./autogen.sh
CC=gcc ./configure --disable-tests --without-mpi \
          --enable-cudart-dynamic  \
          --prefix=/users/palmee/aws-ofi-nccl/install_3/ \
          --with-libfabric=/opt/cray/libfabric/1.15.2.0 --with-cuda=/opt/nvidia/hpc_sdk/Linux_aarch64/24.3/cuda/12.3/ --with-hwloc=/users/palmee/hwloc-2.12.0/install
make
make install


## TEST with gcc all (job 327130) - running

| payload |    busbw   |    algbw   |
| ------: | ---------: | ---------: |
|    1GiB |  91.05GBps |  46.99GBps |
|    2GiB |  91.24GBps |  47.09GBps |
|    4GiB |  91.34GBps |  47.14GBps |

wget https://download.open-mpi.org/release/hwloc/v2.12/hwloc-2.12.0.tar.gz
tar -xvzf hwloc-2.12.0.tar.gz 
cd hwloc-2.12.0
CC=gcc  ./configure --prefix=/users/palmee/hwloc-2.12.0/install
make 
make install

git clone -b v1.14.0 https://github.com/aws/aws-ofi-nccl.git
cd aws-ofi-nccl
mkdir install

./autogen.sh
CC=gcc ./configure --disable-tests --without-mpi \
          --enable-cudart-dynamic  \
          --prefix=/users/palmee/aws-ofi-nccl/install_3/ \
          --with-libfabric=/opt/cray/libfabric/1.15.2.0 --with-cuda=/opt/nvidia/hpc_sdk/Linux_aarch64/24.3/cuda/12.3/ --with-hwloc=/users/palmee/hwloc-2.12.0/install
make
make install

## TEST with local libfabric only for compile (job 327145) - 

| payload |    busbw   |    algbw   |
| ------: | ---------: | ---------: |
|    1GiB |  91.08GBps |  47.01GBps |
|    2GiB |  91.21GBps |  47.08GBps |
|    4GiB |  91.34GBps |  47.14GBps |

git clone -b v1.19.0 https://github.com/ofiwg/libfabric.git
cd libfabric
autoupdate
./autogen.sh
CC=gcc ./configure --prefix=/users/palmee/libfabric/install
make
make install

wget https://download.open-mpi.org/release/hwloc/v2.12/hwloc-2.12.0.tar.gz
tar -xvzf hwloc-2.12.0.tar.gz 
cd hwloc-2.12.0
CC=gcc  ./configure --prefix=/users/palmee/hwloc-2.12.0/install
make 
make install

git clone -b v1.14.0 https://github.com/aws/aws-ofi-nccl.git
cd aws-ofi-nccl
mkdir install

./autogen.sh
CC=gcc ./configure --disable-tests --without-mpi \
          --enable-cudart-dynamic  \
          --prefix=/users/palmee/aws-ofi-nccl/install_4/ \
          --with-libfabric=/users/palmee/libfabric/install --with-cuda=/opt/nvidia/hpc_sdk/Linux_aarch64/24.3/cuda/12.3/ --with-hwloc=/users/palmee/hwloc-2.12.0/install
make
make install

## TEST with local libfabric  compile and job run (job 327161) -


NOT WORKING!!! We need to use the crey lib fabric



## Build plugin in image
export DOCKER_DEFAULT_PLATFORM=linux/amd64
docker run -i -t registry.suse.com/suse/sle15:15.5      
zypper install -y libtool git gcc awk make wget

zypper addrepo https://developer.download.nvidia.com/compute/cuda/repos/opensuse15/x86_64/cuda-opensuse15.repo
zypper refresh
zypper install -y cuda-toolkit-12-3

## Build in Clariden

wget https://download.open-mpi.org/release/hwloc/v2.12/hwloc-2.12.0.tar.gz
tar -xvzf hwloc-2.12.0.tar.gz 
cd hwloc-2.12.0
CC=gcc ./configure --prefix=/users/palmee/hwloc-2.12.0/install
#CC=gcc ./configure
make 
make install


git clone -b v1.14.1 https://github.com/aws/aws-ofi-nccl.git
cd aws-ofi-nccl
mkdir install

./autogen.sh
CC=gcc ./configure --disable-tests --without-mpi \
          --enable-cudart-dynamic  \
          --prefix=/users/palmee/aws-ofi-nccl/install/ \
          --with-libfabric=/opt/cray/libfabric/1.22.0/ \
          --with-cuda=/opt/nvidia/hpc_sdk/Linux_aarch64/24.3/cuda/12.3/ \
          --with-hwloc=/users/palmee/hwloc-2.12.0/install

make 
make install


export LD_LIBRARY_PATH=/opt/cray/libfabric/1.15.2.0/lib64/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_aarch64/24.3/cuda/12.3/lib64/:$LD_LIBRARY_PATH
ld /users/palmee/aws-ofi-nccl/install/lib/libnccl-net.so 


TODO:
consider building an rpm: https://www.redhat.com/en/blog/create-rpm-package