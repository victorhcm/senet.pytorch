#!/bin/bash
#SBATCH -N 1
#SBATCH --qos=high
#SBATCH -c 6
#SBATCH --gres=gpu:2
#SBATCH --nodelist=viper06


# DIRECTORIES
export PROJECTDIR="/home/victorhcmelo/slurm/senet.pytorch"

cd $PROJECTDIR
conda env update 
source activate senet

export PYTHONPATH="$PYTHONPATH:$PROJECTDIR"
# srun python -m cifar.main
# srun python -m cifar.main_pls

# srun python cifar.py --reduction 4
srun python imagenet.py /local/victorhcmelo/imagenet_2017/ --reduction 8 --batch_size 150
