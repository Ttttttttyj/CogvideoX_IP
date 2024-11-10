#! /bin/bash

echo "RUN on `hostname`, CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

environs="WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 LOCAL_WORLD_SIZE=1"

run_cmd="$environs CUDA_VISIBLE_DEVICES=6 python train_video.py --base configs/cogvideox_2b_ip_attention.yaml configs/sft.yaml --seed $RANDOM"

echo ${run_cmd}
eval ${run_cmd}

echo "DONE on `hostname`"