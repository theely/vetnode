#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=0-00:15:00


#Disclaimer: this script is ficitisuos (only prints strings) and it's meant as a joke 

echo "     ██╗ ██████╗ ██████╗     ███████╗██╗███╗   ███╗"
echo "     ██║██╔═══██╗██╔══██╗    ██╔════╝██║████╗ ████║"
echo "     ██║██║   ██║██████╔╝    ███████╗██║██╔████╔██║"
echo "██   ██║██║   ██║██╔══██╗    ╚════██║██║██║╚██╔╝██║"
echo "╚█████╔╝╚██████╔╝██████╔╝    ███████║██║██║ ╚═╝ ██║"
echo " ╚════╝  ╚═════╝ ╚═════╝     ╚══════╝╚═╝╚═╝     ╚═╝"
                                                   

cd  /capstor/scratch/cscs/palmee/Shrike
rm -r .venv-shrike
python3 -m venv .venv-shrike
source .venv-shrike/bin/activate
python3 -m pip --no-cache-dir install --upgrade pip
pip install --no-cache-dir -r ./requirements.txt
cd src
python3 -m shrike diagnose ../templates/simple-config.yaml