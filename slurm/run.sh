#!/bin/bash
#SBATCH -N 1
#SBATCH --qos=high
#SBATCH -c 4
#SBATCH --gres=gpu:1
#rmSBATCH --nodelist=viper05


# DIRECTORIES
export PROJECTDIR="/home/victorhcmelo/slurm/senet.pytorch"

cd $PROJECTDIR
#conda env update 
source activate senet

export PYTHONPATH="$PYTHONPATH:$PROJECTDIR"
# srun python -m cifar.main
# srun python -m cifar.main_pls

srun python cifar.py 
