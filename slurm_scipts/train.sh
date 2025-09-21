#!/bin/sh

#SBATCH	-A jgmakin
#SBATCH -p a100-40gb #,a10,v100  #v100 #a100-40gb
#SBATCH -q normal  #jgmakin-n

#a100-40gb
#a30
# gpu queues: v100, a10, a30 , 

#F|G|I|K|D|B
# jgmakin-n, standby, training
# all valid: I|J|K|N|G|F|H|C|D|B
# very high mem: I|J|K      # 80GB
# High Mem GPUs: I|J|K|N|G    # 40GB
# High Mem GPUs: I|J|K|N|G|F|H|C    # 24GB
# very Fast GPUs: F|G|K
# Fast GPUs: D
# Slow GPUs: E

#SBATCH --nodes=1 
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=40GB
#SBATCH --time=2-04:00:00
#SBATCH --job-name=diff_samples
#SBATCH --output=outputs/%j.out


# activate virtual environment
source ./env_setup.sh

mkdir -p outputs

echo "$PWD"
python ../scripts/train.py $@