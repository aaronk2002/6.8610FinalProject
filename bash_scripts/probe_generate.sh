#!/bin/bash

# Loading Modules in Supercloud, comment out when not using SuperCloud
source /etc/profile
module load anaconda/2023a-pytorch
module load cuda/11.8

# Generate probing tasks
for idx in {4,6,8}
do
    python ../probing/generate_probe_data.py --model ../trained_models/$idx.pth --save ../dataset/$idx-layers-probe.pth
done