#!/bin/bash
#SBATCH	-A jgmakin
#SBATCH -p a100-40gb #,a10,v100  #v100 #a100-40gb
#SBATCH -q normal  #jgmakin-n

#standby
#jgmakin-n
#training
#debug
# --constraint=C|F|G|I|J|K

# F|G|I|K|D|B|H|J|C|N
# High Mem GPUs: C|F|G|I|J|K
# very Fast GPUs: F|K
# Fast GPUs: B|D
# Slow GPUs: E

#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
# --cpus-per-task=2
#SBATCH --mem=250GB
#SBATCH --time=1-23:00:00
#SBATCH --job-name=training_data
#SBATCH --output=outputs/%j.out               # Standard output (%j = job ID)

# activate virtual environment
source ./env_setup.sh

# Print node and rank details
hostname
# Set up environment variables
export MASTER_ADDR=$(hostname -s)         # Set master address to the node's hostname
export MASTER_PORT=29501                  # Port for communication, try: 29500, 29501, etc.

# Output environment variables for logging
echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"

python ../scripts/eeg_data.py


