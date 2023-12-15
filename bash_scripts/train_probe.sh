#!/bin/bash

# Loading Modules in Supercloud
source /etc/profile
module load anaconda/2023a-pytorch
module load cuda/11.8

# Hyperparams
lr=0.0001
epochs=10000
task=composer

# Train
for layer in {4,6,8}
do
    python ../probing/train_probes.py --layers $layer --task $task --lr $lr --epochs $epochs
    python ../probing/evaluate_probes.py --layers $layer --task $task --lr $lr --epochs $epochs
    echo
done