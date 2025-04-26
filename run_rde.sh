#!/bin/bash
root_dir=./datas
tau=0.015 
margin=0.1
noisy_rate=0.2  #0.0 0.2 0.5 0.8
select_ratio=0.3
loss=TAL
DATASET_NAME=MSVD
# CUHK-PEDES ICFG-PEDES RSTPReid MSVD

noisy_file=./noiseindex/${DATASET_NAME}_${noisy_rate}.npy
CUDA_VISIBLE_DEVICES=0 \
    python train_vid.py \
    --noisy_rate $noisy_rate \
    --noisy_file $noisy_file \
    --name RDE \
    --img_aug \
    --txt_aug \
    --batch_size 2 \
    --select_ratio $select_ratio \
    --tau $tau \
    --root_dir $root_dir \
    --output_dir run_logs \
    --margin $margin \
    --dataset_name $DATASET_NAME \
    --video_frame_rate 1 \
    --loss_names ${loss}+sr${select_ratio}_tau${tau}_margin${margin}_n${noisy_rate}  \
    --num_epoch 60 
 