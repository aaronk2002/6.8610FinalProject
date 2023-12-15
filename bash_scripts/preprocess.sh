#!/bin/bash

# For Loop
for seed in {2004,2006,2008,2009,2011,2013,2014,2015,2017,2018}
do
   echo $seed
   python ../preprocess.py ../dataset/maestro-v3.0.0/$seed ../dataset/processed
done