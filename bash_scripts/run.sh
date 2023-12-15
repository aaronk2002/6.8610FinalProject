#!/bin/bash

# Loading Modules in Supercloud, comment out when not using SuperCloud
source /etc/profile
module load anaconda/2023a-pytorch
module load cuda/11.8

python ../train.py -c config/addison.yml -m config
python ../generate.py -c config/generate.yml -m config