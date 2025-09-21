#!/bin/bash

# Environment setup for Slurm jobs
echo "Hostname: $(hostname)"
echo "Allocated memory per node: $((${SLURM_MEM_PER_NODE} / 1024)) GB"
echo "Number of GPUs: $SLURM_GPUS_PER_NODE"
export NUMBA_DISABLE_INTEL_SVML=1
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "GPU Info:"
nvidia-smi

module purge
module load external
module load conda 
conda activate DAPS
# 
# conda activate /home/ahmedb/.conda/envs/cent7/2020.11-py38/wav2letter
