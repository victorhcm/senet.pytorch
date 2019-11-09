#!/bin/bash
#SBATCH -N 1
#SBATCH --qos=high
#SBATCH -c 2
#SBATCH --gres=gpu:1
#SBATCH --nodelist=viper06

cd /ssd/victorhcmelo/imagenet/ILSVRC/Data/CLS-LOC/val
wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
