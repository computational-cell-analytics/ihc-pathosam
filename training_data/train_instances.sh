#! /bin/bash
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem 128G
#SBATCH -t 720
#SBATCH -p grete:shared
#SBATCH -G A100:1

source ~/.bashrc
micromamba activate sam
python train_instances.py $@ 
