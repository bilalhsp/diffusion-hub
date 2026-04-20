#!/bin/sh

#SBATCH	-A jgmakin
#SBATCH -p a100-40gb #,a10,v100  #v100 #a100-40gb
#SBATCH -q normal  #jgmakin-n

#SBATCH --nodes=1 
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=40GB
#SBATCH --time=2-04:00:00
#SBATCH --job-name=download_data
#SBATCH --output=outputs/%j.out

# mkdir -p /scratch/gilbreth/ahmedb/data/ljspeech
# wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2 -P /scratch/gilbreth/ahmedb/data/ljspeech
# tar -xjf /scratch/gilbreth/ahmedb/data/ljspeech/LJSpeech-1.1.tar.bz2 -C /scratch/gilbreth/ahmedb/data/ljspeech
# echo "LJSpeech downloaded and extracted successfully to /path/to/dir"


# activate virtual environment
source ./env_setup.sh

cp /home/ahmedb/projects/hifi-gan/LJSpeech-1.1/training.txt /scratch/gilbreth/ahmedb/data/ljspeech/LJSpeech-1.1/
cp /home/ahmedb/projects/hifi-gan/LJSpeech-1.1/validation.txt /scratch/gilbreth/ahmedb/data/ljspeech/LJSpeech-1.1/
echo "Train and validation files copied successfully"


# echo "Creatng spectrograms..."
# python ../scripts/hifi_gan_ft_dataset.py $@
