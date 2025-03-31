#!/bin/bash

#SBATCH --qos=normal
#SBATCH --partition=rtx6000
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=24G
#SBATCH --time=16:00:00
#SBATCH --output=./logs_infer.txt
#SBATCH --open-mod=apend
#SBATCH --job-name=infer5task
#SBATCH --ntasks=1

source /scratch/ssd004/scratch/hana/midienv/bin/activate
srun --mem=24G python inference_cnn.py &
wait