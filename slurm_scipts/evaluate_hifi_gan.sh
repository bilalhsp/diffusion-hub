#!/bin/sh

#SBATCH	-A jgmakin
#SBATCH -p a100-40gb #,a10,v100  #v100 #a100-40gb
#SBATCH -q normal  #jgmakin-n

#SBATCH --nodes=1 
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=40GB
#SBATCH --time=1-04:00:00
#SBATCH --job-name=training
#SBATCH --output=outputs/%j.out

# activate virtual environment
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
conda activate /depot/jgmakin/data/bilal/env/speech

cd /home/ahmedb/projects/hifi-gan

python -u compare_checkpoints.py 

