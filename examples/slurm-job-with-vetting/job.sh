#!/bin/bash

#SBATCH --nodes=6
#SBATCH --time=0-00:15:00
#SBATCH --account=a-csstaff

#---------------------------------------------------------                                               
#Parameters
#---------------------------------------------------------

# Set the exact number of nodes required to run the job.
# You can allocate (#SBATCH --nodes=xy) more nodes than 
# required to account for non healthy ones. 
REQUIRED_NODES=4

# The application/command you would like to run on the
# vetted nodes.
MAIN_JOB_COMMAND=hostname
#---------------------------------------------------------

echo "███████╗ █████╗ ███╗   ██╗██╗████████╗██╗   ██╗"
echo "██╔════╝██╔══██╗████╗  ██║██║╚══██╔══╝╚██╗ ██╔╝"
echo "███████╗███████║██╔██╗ ██║██║   ██║    ╚████╔╝ "
echo "╚════██║██╔══██║██║╚██╗██║██║   ██║     ╚██╔╝  "
echo "███████║██║  ██║██║ ╚████║██║   ██║      ██║   "
echo "╚══════╝╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝   ╚═╝      ╚═╝   "


# Set-up environment and node vetting cli
WORK_DIR="vetnode-$SLURM_JOB_ID"
mkdir $WORK_DIR
cd $WORK_DIR

touch "./results.txt"

python3.11 -m venv .venv
source .venv/bin/activate
pip install --no-cache-dir vetnode


#Add CUDA
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/nvidia/hpc_sdk/Linux_aarch64/24.3/cuda/12.3/lib64/

#Setup node vetting on main node
vetnode setup ../examples/slurm-job-with-vetting/simple-config.yaml &>> ./results.txt


# Run nodes vetting
srun vetnode diagnose ../examples/slurm-job-with-vetting/simple-config.yaml &>> ./results.txt

# Extract node lists
grep '^Cordon:' ./results.txt | awk '{print $2}' > ./cordoned-nodes.txt
grep '^Vetted:' ./results.txt | awk '{print $2}' > ./vetted-nodes.txt

#Run on healthy nodes only
if [ $(wc -l < ./vetted-nodes.txt) -ge $REQUIRED_NODES ]; then
    srun -N $REQUIRED_NODES --exclude=./cordoned-nodes.txt $MAIN_JOB_COMMAND
else
    echo "Job aborted!"
    echo "Reason: too few vetted nodes."
fi
