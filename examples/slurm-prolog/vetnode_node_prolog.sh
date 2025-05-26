#!/bin/bash

if [[ "${SLURM_JOB_CONSTRAINTS}x" =~ "vetnode" ]]; then
    
     echo "Running Vetnode on host: $(hostname)"
    
    # Initialize
    . /etc/slurm/utils/kafka_logger
    export PYTHONDONTWRITEBYTECODE=1

    # Extract vetnode variables
    VETNODE_HOME=$(echo "${SLURM_JOB_COMMENT}" | grep -oP 'VETNODE_HOME=\S+' | cut -d= -f2)

    # Set default vetnode home location if not set
    if [ -z "${VETNODE_HOME}" ]; then
        VETNODE_HOME="/opt/cscs/vetnode"
    fi

    # Verify configuration can be loaded
    if [ ! -f "${VETNODE_CONFIGURATION}/config.yaml" ]; then
        reason="Vetnode configuration not found!"
        echo "$reason"
        publish_error "Node $(hostname) will skip node vetting with reason: ${reason}" "prolog"
        exit 0
    fi

    # Run Vetnode
    source $VETNODE_HOME/.venv/bin/activate
    timeout 10s vetnode diagnose ${VETNODE_HOME}/config.yaml

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

