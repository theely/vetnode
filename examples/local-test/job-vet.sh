#!/bin/bash

#SBATCH --nodes=2
#SBATCH --time=0-00:15:00
#SBATCH --account=a-csstaff

#---------------------------------------------------------                                               
#Parameters
#---------------------------------------------------------

echo "██╗   ██╗███████╗████████╗███╗   ██╗ ██████╗ ██████╗ ███████╗"
echo "██║   ██║██╔════╝╚══██╔══╝████╗  ██║██╔═══██╗██╔══██╗██╔════╝"
echo "██║   ██║█████╗     ██║   ██╔██╗ ██║██║   ██║██║  ██║█████╗  "
echo "╚██╗ ██╔╝██╔══╝     ██║   ██║╚██╗██║██║   ██║██║  ██║██╔══╝  "
echo " ╚████╔╝ ███████╗   ██║   ██║ ╚████║╚██████╔╝██████╔╝███████╗"
echo "  ╚═══╝  ╚══════╝   ╚═╝   ╚═╝  ╚═══╝ ╚═════╝ ╚═════╝ ╚══════╝"
                                                             

# Set-up environment and node vetting cli
WORK_DIR="vetnode-$SLURM_JOB_ID"

# Download vetnode source code
git clone https://github.com/theely/vetnode.git $WORK_DIR
cd $WORK_DIR

mkdir aws-ofi-nccl
mkdir aws-ofi-nccl/lib
arch=$(uname -m)
curl -o ./aws-ofi-nccl/lib/libnccl-net.so https://jfrog.svc.cscs.ch/artifactory/aws-ofi-nccl-gen-dev/v1.9.2-aws-cf6f657/${arch}/SLES/15.5/cuda12/lib/libnccl-net.so
export PATH_PLUGIN=$(pwd)/aws-ofi-nccl


python3.11 -m venv .venv
source .venv/bin/activate
python -m pip --no-cache-dir install --upgrade pip
pip install --no-cache-dir -r ./requirements.txt

#Add CUDA
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/nvidia/hpc_sdk/Linux_aarch64/24.3/cuda/12.3/lib64/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/nvidia/hpc_sdk/Linux_aarch64/2024/comm_libs/nccl/lib/

#export LD_LIBRARY_PATH=/opt/cray/libfabric/1.15.2.0/lib64/:$LD_LIBRARY_PATH
#export LD_LIBRARY_PATH=$PATH_PLUGIN/lib/:$LD_LIBRARY_PATH
#export LD_PRELOAD=$PATH_PLUGIN/lib/libnccl-net.so 

export CXI_FORK_SAFE="1"
export CXI_FORK_SAFE_HP="1"
export FI_CXI_DISABLE_CQ_HUGETLB="1"
export NCCL_CROSS_NIC="1"
export NCCL_DEBUG="Info"
export NCCL_NET_GDR_LEVEL="PHB"
export FI_CXI_DISABLE_HOST_REGISTER="1"
export FI_MR_CACHE_MONITOR="userfaultfd"

cd src

#Setup node vetting on main node
python -m vetnode  setup ../examples/local-test/config.yaml

# Run nodes vetting
srun python -m vetnode  diagnose ../examples/local-test/config.yaml

