#!/bin/bash

# Loading Modules in Supercloud
source /etc/profile
module load anaconda/2023a-pytorch
module load cuda/11.8

# Evaluate
for layer in {4,6,8}
do
    echo $layer layers
    python ../evaluation.py --layers $layer --N 100 --M 10
done