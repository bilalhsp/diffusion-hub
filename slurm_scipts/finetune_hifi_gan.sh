#!/bin/sh

#SBATCH	-A jgmakin
#SBATCH -p a100-40gb #,a10,v100  #v100 #a100-40gb
#SBATCH -q normal  #jgmakin-n

#SBATCH --nodes=1 
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=40GB
#SBATCH --time=5-00:00:00
#SBATCH --job-name=training
#SBATCH --output=outputs/%j.out

# activate virtual environment
source ./env_setup.sh

cd /home/ahmedb/projects/hifi-gan

python train.py \
    --input_wavs_dir /scratch/gilbreth/ahmedb/data/ljspeech/LJSpeech-1.1/wavs \
    --input_mels_dir /scratch/gilbreth/ahmedb/data/ljspeech/spectrograms \
    --input_training_file /scratch/gilbreth/ahmedb/data/ljspeech/LJSpeech-1.1/training.txt \
    --input_validation_file /scratch/gilbreth/ahmedb/data/ljspeech/LJSpeech-1.1/validation.txt \
    --checkpoint_path /scratch/gilbreth/ahmedb/data/ljspeech/checkpoints \
    --config /scratch/gilbreth/ahmedb/data/ljspeech/checkpoints/config.json \
    --training_epochs 3200 \
    --fine_tuning True
