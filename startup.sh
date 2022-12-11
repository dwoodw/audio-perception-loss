#!/bin/bash

#set gpu
export CUDA_VISIBLE_DEVICES=0



#cd "/home/dwoodw19/thesis/BSS_metric"
#cd "/home/dwoodward/masters/audio-perception-loss"
cwd=$(pwd)

cd "$cwd"
mkdir -p userlogs
python -u demo.py with cfg.baseline >> "$cwd/userlogs/log.txt" &

