#!/bin/bash
#SBATCH --job-name=EfficientNet
#SBATCH --output=vasp.out
#SBATCH --error=vasp.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=v.pathak@ufl.edu
#SBATCH --nodes=1                    
#SBATCH --ntasks=1                   
#SBATCH --cpus-per-task=10            
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=60gb
#SBATCH --time=48:00:00

echo "Date      = $(date)"
echo "host      = $(hostname -s)"
echo "Directory = $(pwd)"

module purge
ml pytorch/1.8.1

T1=$(date +%s)
python BYOL_SIIM-ISIC_Efficientnetb5.py
T2=$(date +%s)

ELAPSED=$((T2 - T1))
echo "Elapsed Time = $ELAPSED"

