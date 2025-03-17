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
touch cordoned-nodes.txt

srun python -m shrike diagnose ../templates/simple-config.yaml >> cordoned-nodes.txt