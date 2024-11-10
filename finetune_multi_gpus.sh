#! /bin/bash

echo "RUN on $(hostname), CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

NUM_GPUS=3

run_cmd="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True torchrun --standalone --nproc_per_node=$NUM_GPUS train_video.py --base configs/cogvideox_2b_ip_attention.yaml configs/sft.yaml --seed $RANDOM"

echo ${run_cmd}
eval ${run_cmd}

echo "DONE on `hostname`"