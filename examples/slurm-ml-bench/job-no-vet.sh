#!/bin/bash

#SBATCH --nodes=256
#SBATCH --time=0-00:15:00
#SBATCH --account=a-csstaff

#---------------------------------------------------------                                               
#Parameters
#---------------------------------------------------------

# Set the exact number of nodes required to run the job.
# You can allocate (#SBATCH --nodes=xy) more nodes than 
# required to account for non healthy ones. 
REQUIRED_NODES=256

#---------------------------------------------------------

echo "██╗   ██╗███████╗████████╗███╗   ██╗ ██████╗ ██████╗ ███████╗"
echo "██║   ██║██╔════╝╚══██╔══╝████╗  ██║██╔═══██╗██╔══██╗██╔════╝"
echo "██║   ██║█████╗     ██║   ██╔██╗ ██║██║   ██║██║  ██║█████╗  "
echo "╚██╗ ██╔╝██╔══╝     ██║   ██║╚██╗██║██║   ██║██║  ██║██╔══╝  "
echo " ╚████╔╝ ███████╗   ██║   ██║ ╚████║╚██████╔╝██████╔╝███████╗"
echo "  ╚═══╝  ╚══════╝   ╚═╝   ╚═╝  ╚═══╝ ╚═════╝ ╚═════╝ ╚══════╝"
                                                             

# Set-up environment and node vetting cli
WORK_DIR="/users/palmee/vetnode_bench/$SLURM_JOB_ID"
mkdir $WORK_DIR
cd $WORK_DIR


#python3.11 -m venv /users/palmee/vetnode_bench/.venv
source /users/palmee/vetnode_bench/.venv/bin/activate
#pip install --no-cache-dir vetnode


curl -o all_reduce_bench.py https://raw.githubusercontent.com/theely/vetnode/refs/heads/main/examples/slurm-ml-vetting/all_reduce_bench.py


#Add CUDA
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/nvidia/hpc_sdk/Linux_aarch64/24.3/cuda/12.3/lib64/


export PATH_PLUGIN=/users/palmee/aws-ofi-nccl/install_4

# Activate AWS NCCL plugin
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_aarch64/24.3/cuda/12.3/lib64/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/cray/libfabric/1.15.2.0/lib64/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$PATH_PLUGIN/lib/:$LD_LIBRARY_PATH
export LD_PRELOAD=$PATH_PLUGIN/lib/libnccl-net.so 
export CXI_FORK_SAFE="1"
export CXI_FORK_SAFE_HP="1"
export FI_CXI_DISABLE_CQ_HUGETLB="1"
export NCCL_CROSS_NIC="1"
export NCCL_DEBUG="Info"
export NCCL_NET_GDR_LEVEL="PHB"
export FI_CXI_DISABLE_HOST_REGISTER="1"
export FI_MR_CACHE_MONITOR="userfaultfd"


srun -N $REQUIRED_NODES --tasks-per-node=1 python -u -m torch.distributed.run --nproc_per_node=4 \
--nnodes $REQUIRED_NODES --rdzv_endpoint $(hostname):6000 --rdzv_backend \
c10d all_reduce_bench.py

