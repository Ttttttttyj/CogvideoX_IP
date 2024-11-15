#! /bin/bash

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

environs="WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 LOCAL_WORLD_SIZE=1"

run_cmd="$environs python sample_video_ip_attention.py --base configs/cogvideox_2b_ip_attention.yaml configs/inference.yaml --seed $RANDOM"

echo ${run_cmd}
eval ${run_cmd}

echo "DONE on `hostname`"