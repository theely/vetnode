#!/bin/bash

if [[ "${SLURM_JOB_CONSTRAINTS}x" =~ "vetnode" ]]; then
    
     echo "Running Vetnode on host: $(hostname)"
    
    # Initialize
    . /etc/slurm/utils/kafka_logger
    export PYTHONDONTWRITEBYTECODE=1

    VETNODE_HOME=$(echo "${SLURM_JOB_COMMENT}" | grep -oP 'VETNODE_HOME=\S+' | cut -d= -f2)
    VETNODE_CONFIGURATION=$(echo "${SLURM_JOB_COMMENT}" | grep -oP 'VETNODE_CONFIGURATION=\S+' | cut -d= -f2)

    # Verify configuration can be loaded
    if [ ! -f "${VETNODE_CONFIGURATION}" ]; then
        reason="Vetnode configuration not found!"
        echo "$reason"
        publish_error "Node $(hostname) will skip node vetting with reason: ${reason}" "prolog"
        exit 0
    fi

    # Verify vetnode installation is present
    if [ -z "${VETNODE_HOME}" ]; then
        reason="Unable to find Vetnode installation!"
        echo "$reason"
        publish_error "Node $(hostname) will skip node vetting with reason: ${reason}" "prolog"
        exit 0
    fi

    #Search for CUDA installation
    CUDA_HOME=$(dirname $(find /opt/nvidia/hpc_sdk/ -name "libnvrtc.so" | grep -e "cuda/12.3" -e "cuda/11.6" |  head -n 1))
    if [ ! -d "$CUDA_HOME" ]; then
        reason="unable to find CUDA installation path"
        echo "$reason"
        publish_error "Node $(hostname) will not be allocated with reason: ${reason}" "prolog"
        exit 1
    fi
    
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME


    # Run Vetnode
    source $VETNODE_HOME/.venv/bin/activate
    timeout 10s vetnode diagnose ${VETNODE_CONFIGURATION}

    rc=$?
    if [ $rc == 124 ]; then
        reason="Vetnode diagnose did timeout"
        echo "$reason"
        publish_error "Node $(hostname) will not be allocated with reason: ${reason}" "prolog"
        exit 1
    fi
    if [ $rc != 0 ]; then
        reason="Vetnode tests failed"
        echo "$reason"
        publish_error "Node $(hostname) will not be allocated with reason: ${reason}" "prolog"
        exit 1
    fi
fi

exit 0
