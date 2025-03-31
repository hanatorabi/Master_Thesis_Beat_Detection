#!/bin/bash
#SBATCH --qos=m
#SBATCH --partition=rtx6000
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=./logs.txt
#SBATCH --open-mod=apend
#SBATCH --job-name=5task


source /scratch/ssd004/scratch/hana/midienv/bin/activate
srun --mem=24G python traintest.py &
wait