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
                                                             

# Activate AWS NCCL plugin
export PATH_PLUGIN=$(pwd)/aws-ofi-nccl
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/nvidia/hpc_sdk/Linux_aarch64/24.3/cuda/12.3/lib64/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/nvidia/hpc_sdk/Linux_aarch64/2024/comm_libs/nccl/lib/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/cray/libfabric/1.22.0/lib64/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PATH_PLUGIN/lib/
export LD_PRELOAD=$PATH_PLUGIN/lib/libnccl-net.so 

# Official flags https://eth-cscs.github.io/cscs-docs/software/communication/nccl/
export NCCL_NET_PLUGIN="ofi"  # with uenv export NCCL_NET="AWS Libfabric"
export NCCL_NET_GDR_LEVEL="PHB"
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_CXI_DEFAULT_TX_SIZE=32768
export FI_CXI_DISABLE_HOST_REGISTER=1
export FI_CXI_RX_MATCH_MODE=software
export FI_MR_CACHE_MONITOR="userfaultfd"
export MPICH_GPU_SUPPORT_ENABLED=0

# Other flags 
# export CXI_FORK_SAFE="1"
# export CXI_FORK_SAFE_HP="1"
# export FI_CXI_DISABLE_CQ_HUGETLB="1"
export NCCL_CROSS_NIC="1"
export NCCL_DEBUG="Info"


source .venv/bin/activate

cd src

#Setup node vetting on main node
python -m vetnode  setup ../examples/local-test/config.yaml

# Run nodes vetting
srun  python -m vetnode  diagnose ../examples/local-test/config.yaml

