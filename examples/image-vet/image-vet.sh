#!/bin/bash

#SBATCH --nodes=2
#SBATCH --time=0-00:15:00
#SBATCH --account=csstaff



if [ -z "$1" ]; then
    echo "❌ Error: missing required image argument."
    echo
    echo "Usage:"
    echo "  sbatch image-vet.sh <ARG1>"
    echo
    echo "Example:"
    echo "  sbatch image-vet.sh nvcr.io#nvidia/pytorch:25.12-py3"
    exit 1
fi

IMAGE_NAME="$1"

echo "[image-vet] Evaluating Image: $IMAGE_NAME"


export ENV_FILE="/tmp/env.toml"
cat > "$ENV_FILE" <<- EOF

image = "${IMAGE_NAME}"

mounts = [
    "/users/${USER}",
    "/capstor/",
    "/iopsstor/",
    "/tmp",
    # Options for simulated built-in OFI NCCL plugin
    # "/opt/cscs/aws-ofi-ccl-plugin/cuda-dl/libnccl-net.so:/usr/lib/libnccl-net-oficscs.so",
    # "/opt/cscs/aws-ofi-ccl-plugin/cuda-dl/libnccl-net.so:/opt/hpcx/nccl_rdma_sharp_plugin/lib/libnccl-net.so" 
]

writable = true

# Options for built-in OFI NCCL plugin
[env]
NCCL_NET_PLUGIN = "oficscs"
NCCL_NET = "AWS Libfabric"
NCCL_CROSS_NIC = "1"
NCCL_NET_GDR_LEVEL = "PHB"
OFI_NCCL_DISABLE_DMABUF = "1"

# Options for OFI NCCL plugin via hooks
#[annotations]
#com.hooks.aws_ofi_nccl.enabled = "true"
#com.hooks.aws_ofi_nccl.variant = "cuda13"

EOF


cat > config.yaml <<- EOF
name: Image Vetting
scheduler: slurm
evals:
- name: CudaKernel
  type: vetnode.evaluations.cuda_eval.CUDAEval
  cuda_home: /usr/local/cuda 
  requirements:
    - cuda-python
    - numpy
- name: NCCL-Low-Level-Internode
  type: vetnode.evaluations.nccl_lib_eval.NcclLibEval
  topology: internode
  scheduler: slurm
  payload: 8 GB
  method: allreduce
  min_bandwidth: 300 GB/s
  warmup:
    payload: 256 MB
    runs: 2
  requirements: ["cuda-python","numpy"]
- name: NCCL-Low-Level-Intranode
  type: vetnode.evaluations.nccl_lib_eval.NcclLibEval
  topology: intranode
  scheduler: slurm
  payload: 8 GB
  method: allreduce
  min_bandwidth: 18 GB/s
  warmup:
    payload: 256 MB
    runs: 2
  requirements: ["cuda-python","numpy"]
- name: NCCL-Low-Level-Full-Topology
  type: vetnode.evaluations.nccl_lib_eval.NcclLibEval
  scheduler: slurm
  payload: 8 GB
  method: allreduce
  min_bandwidth: 80 GB/s
  warmup:
    payload: 256 MB
    runs: 2
  requirements: ["cuda-python","numpy"]
- name: NCCL-PyTorch
  type: vetnode.evaluations.nccl_pytorch_eval.NcclPytorchEval
  scheduler: slurm
  payload: 8 GB
  method: allreduce
  min_bandwidth: 125 GB/s
  warmup:
    payload: 256 MB
    runs: 2
  requirements:
      - ['torch', '--index-url', 'https://download.pytorch.org/whl/cu130']
      - numpy
EOF

sbcast config.yaml /tmp/config.yaml

srun -N ${SLURM_JOB_NUM_NODES} --tasks-per-node=1 -u --environment=${ENV_FILE} --container-writable bash -c '

    echo "[image-vet] Set-up vetnode on $(hostname)..." 
    cd /tmp
    python -m venv --system-site-packages .venv
    source .venv/bin/activate
    pip install --no-cache-dir vetnode
    vetnode setup /tmp/config.yaml 
'

srun -N ${SLURM_JOB_NUM_NODES} --tasks-per-node=4 -u --environment=${ENV_FILE} --container-writable bash -c '
    
    #Enable logging
    # export NCCL_DEBUG=INFO
    # export NCCL_DEBUG_SUBSYS=INIT,NET	

    echo "[image-vet] diagnose"
    cd /tmp
    source .venv/bin/activate
    vetnode diagnose /tmp/config.yaml
'