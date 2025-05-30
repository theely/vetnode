#!/bin/bash

#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
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
                                                             


export PATH_PLUGIN=$(pwd)/aws-ofi-nccl
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/nvidia/hpc_sdk/Linux_aarch64/24.3/cuda/12.3/lib64/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/nvidia/hpc_sdk/Linux_aarch64/2024/comm_libs/nccl/lib/

export CXI_FORK_SAFE="1"
export CXI_FORK_SAFE_HP="1"
export FI_CXI_DISABLE_CQ_HUGETLB="1"
export NCCL_CROSS_NIC="1"
export NCCL_DEBUG="Info"
export NCCL_NET_GDR_LEVEL="PHB"
export FI_CXI_DISABLE_HOST_REGISTER="1"
export FI_MR_CACHE_MONITOR="userfaultfd"

source .venv/bin/activate

cd src

echo "=== SETUP ==="

#Setup node vetting on main node
python -m vetnode  setup ../examples/local-test/config.yaml

echo "=== VETTING ==="

# Run nodes vetting
srun  python -m vetnode  diagnose ../examples/local-test/config.yaml

