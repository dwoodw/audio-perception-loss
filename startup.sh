#!/bin/bash

#set gpu
export CUDA_VISIBLE_DEVICES=0

# set working directory
cd 
cd "/home/dwoodw19/thesis/BSS_metric"
cwd=$(pwd)

cd "$cwd"
mkdir -p userlogs
python -u demo.py with cfg.AAUmachine >> "$cwd/userlogs/log.txt" &

