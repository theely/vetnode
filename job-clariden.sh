#!/bin/bash

#SBATCH --nodes=4
#SBATCH --time=0-00:15:00
#SBATCH --account=a-csstaff

#Disclaimer: this script is ficitisuos (only prints strings) and it's meant as a joke 

echo "     ██╗ ██████╗ ██████╗     ███████╗██╗███╗   ███╗"
echo "     ██║██╔═══██╗██╔══██╗    ██╔════╝██║████╗ ████║"
echo "     ██║██║   ██║██████╔╝    ███████╗██║██╔████╔██║"
echo "██   ██║██║   ██║██╔══██╗    ╚════██║██║██║╚██╔╝██║"
echo "╚█████╔╝╚██████╔╝██████╔╝    ███████║██║██║ ╚═╝ ██║"
echo " ╚════╝  ╚═════╝ ╚═════╝     ╚══════╝╚═╝╚═╝     ╚═╝"
                                                   

rm -rf shrike-deploy
git clone https://github.com/theely/shrike.git shrike-deploy
cd shrike-deploy
python3.11 -m venv .venv-shrike
source .venv-shrike/bin/activate
python -m pip --no-cache-dir install --upgrade pip
pip install --no-cache-dir -r ./requirements.txt
cd src
rm /users/palmee/shrike-deploy/sanity-results.txt
touch /users/palmee/shrike-deploy/sanity-results.txt

srun python -m shrike diagnose ../templates/simple-config.yaml >> /users/palmee/shrike-deploy/sanity-results.txt

# Exclude approach
# grep '^Cordon:' /users/palmee/shrike-deploy/sanity-results.txt | awk '{print $2}' | paste -sd ',' > cordoned-nodes.txt
# srun -N $(grep -c '^Vetted:' /users/palmee/shrike-deploy/sanity-results.txt) --exclude=$(cat cordoned-nodes.txt) hostname

# Free-up approach
grep '^Vetted:' /users/palmee/shrike-deploy/sanity-results.txt | awk '{print $2}' > vetted-nodes.txt
salloc -N $(wc -l < vetted-nodes.txt) --nodefile=vetted-nodes.txt
srun hostname