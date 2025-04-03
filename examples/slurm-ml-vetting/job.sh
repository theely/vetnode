#!/bin/bash

#SBATCH --nodes=3
#SBATCH --time=0-00:15:00
#SBATCH --account=a-csstaff

#---------------------------------------------------------                                               
#Parameters
#---------------------------------------------------------

# Set the exact number of nodes required to run the job.
# You can allocate (#SBATCH --nodes=xy) more nodes than 
# required to account for non healthy ones. 
REQUIRED_NODES=2

# The application/command you would like to run on the
# vetted nodes.
MAIN_JOB_COMMAND=python -u -m torch.distributed.run --nproc_per_node=4 \
                --nnodes $REQUIRED_NODES --rdzv_endpoint $(hostname):6000 --rdzv_backend \
                c10d all_reduce_bench.py
#---------------------------------------------------------

echo "██╗   ██╗███████╗████████╗███╗   ██╗ ██████╗ ██████╗ ███████╗"
echo "██║   ██║██╔════╝╚══██╔══╝████╗  ██║██╔═══██╗██╔══██╗██╔════╝"
echo "██║   ██║█████╗     ██║   ██╔██╗ ██║██║   ██║██║  ██║█████╗  "
echo "╚██╗ ██╔╝██╔══╝     ██║   ██║╚██╗██║██║   ██║██║  ██║██╔══╝  "
echo " ╚████╔╝ ███████╗   ██║   ██║ ╚████║╚██████╔╝██████╔╝███████╗"
echo "  ╚═══╝  ╚══════╝   ╚═╝   ╚═╝  ╚═══╝ ╚═════╝ ╚═════╝ ╚══════╝"
                                                             

# Set-up environment and node vetting cli
WORK_DIR="vetnode-$SLURM_JOB_ID"
mkdir $WORK_DIR
cd $WORK_DIR

# Download example configuration
curl -o config.yaml https://raw.githubusercontent.com/theely/vetnode/refs/heads/main/examples/slurm-ml-vetting/config.yaml
touch "./results.txt"

python3.11 -m venv .venv
source .venv/bin/activate
pip install --no-cache-dir vetnode

#Add CUDA
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/nvidia/hpc_sdk/Linux_aarch64/24.3/cuda/12.3/lib64/

#Setup node vetting on main node
vetnode setup ./config.yaml &>> ./results.txt

# Run nodes vetting
srun vetnode diagnose ./config.yaml &>> ./results.txt

# Extract node lists
grep '^Cordon:' ./results.txt | awk '{print $2}' > ./cordoned-nodes.txt
grep '^Vetted:' ./results.txt | awk '{print $2}' > ./vetted-nodes.txt


#Run on healthy nodes only
if [ $(wc -l < ./vetted-nodes.txt) -ge $REQUIRED_NODES ]; then
    
    #srun -N $REQUIRED_NODES --exclude=./cordoned-nodes.txt $MAIN_JOB_COMMAND
    
    pip install torch --index-url https://download.pytorch.org/whl/cu126
    curl -o all_reduce_bench.py https://raw.githubusercontent.com/theely/vetnode/refs/heads/main/examples/slurm-ml-vetting/all_reduce_bench.py
    
    mkdir aws-ofi-nccl
    mkdir aws-ofi-nccl/lib
    arch=$(uname -m)
    curl -o ./aws-ofi-nccl/lib/libnccl-net.so https://jfrog.svc.cscs.ch/artifactory/aws-ofi-nccl-gen-dev/v1.9.2-aws-cf6f657/${arch}/SLES/15.5/cuda12/lib/libnccl-net.so
    export PATH_PLUGIN=$(pwd)/aws-ofi-nccl

    # Activate AWS NCCL plugin
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


    EXCLUDE_ARG=""
    if [[ -s cordoned-nodes.txt ]]; then
        EXCLUDE_ARG="--exclude=./cordoned-nodes.txt"
    fi

    srun -N $REQUIRED_NODES $EXCLUDE_ARG --tasks-per-node=1 python -u -m torch.distributed.run --nproc_per_node=4 \
                --nnodes $REQUIRED_NODES --rdzv_endpoint $(hostname):6000 --rdzv_backend \
                c10d all_reduce_bench.py

else
    echo "Job aborted!"
    echo "Reason: too few vetted nodes."
fi
