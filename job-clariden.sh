#!/bin/bash

#SBATCH --nodes=4
#SBATCH --time=0-00:15:00
#SBATCH --account=a-csstaff


echo "███████╗ █████╗ ███╗   ██╗██╗████████╗██╗   ██╗"
echo "██╔════╝██╔══██╗████╗  ██║██║╚══██╔══╝╚██╗ ██╔╝"
echo "███████╗███████║██╔██╗ ██║██║   ██║    ╚████╔╝ "
echo "╚════██║██╔══██║██║╚██╗██║██║   ██║     ╚██╔╝  "
echo "███████║██║  ██║██║ ╚████║██║   ██║      ██║   "
echo "╚══════╝╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝   ╚═╝      ╚═╝   "
                                               
                                                   
# Set-up environment
rm -rf shrike-deploy
git clone https://github.com/theely/shrike.git shrike-deploy
cd shrike-deploy
python3.11 -m venv .venv-shrike
source .venv-shrike/bin/activate
python -m pip --no-cache-dir install --upgrade pip
pip install --no-cache-dir -r ./requirements.txt
cd src
touch sanity-results.txt

srun python -m shrike diagnose ../templates/simple-config.yaml >> sanity-results.txt

# Extract node lists
grep '^Cordon:' sanity-results.txt | awk '{print $2}' > cordoned-nodes.txt
grep '^Vetted:' sanity-results.txt | awk '{print $2}' > vetted-nodes.txt

#Run on healthy nodes only
srun -N $(wc -l < vetted-nodes.txt) --exclude=./cordoned-nodes.txt hostname