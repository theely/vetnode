#!/bin/bash

if [[ "${SLURM_JOB_CONSTRAINTS}x" =~ "vetnode" ]]; then
    
     echo "Running Vetnode on host: $(hostname)"
    
    # Initialize
    . /etc/slurm/utils/kafka_logger
    export PYTHONDONTWRITEBYTECODE=1
    # Clariden
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/nvidia/hpc_sdk/Linux_aarch64/24.3/cuda/12.3/lib64/
    # Zinal 
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/nvidia/hpc_sdk/Linux_x86_64/22.3/cuda/lib64/
    
    # Search vetnode installation
    
    #TODO: replace vetnode intallation locations with a common cscs path
    LOCATIONS=(
        "/capstor/scratch/cscs/palmee/vetnode/"
        "/scratch/shared/users/palmee/vetnode/"
    )

    VETNODE_DIR=""

    for folder in "${LOCATIONS[@]}"; do
        if [ -d "$folder" ]; then
            VETNODE_DIR="$folder"
            break
        fi
    done

    if [ -z "${VETNODE_DIR}" ]; then
        reason="Unable to find Vetnode installation"
        echo "$reason"
        publish_error "Node $(hostname) will not be allocated with reason: ${reason}" "prolog"
        exit 1
    fi

    # Run Vetnode
    source $VETNODE_DIR/.venv/bin/activate
    vetnode diagnose $VETNODE_DIR/config.yaml

    rc=$?
    if [ $rc != 0 ]; then
        reason="Vetnode tests failed"
        echo "$reason"
        publish_error "Node $(hostname) will not be allocated with reason: ${reason}" "prolog"
        exit 1
    fi
fi

exit 0
