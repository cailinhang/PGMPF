#!/bin/bash


change_layer_end_for_different_structure(){
resnet110 324
resnet56 162
resnet32 90
resnet20 54
}

pruning(){
CUDA_VISIBLE_DEVICES=$1 python pruning_resnet_soft_hard_gradient_decay_channelwise.py   --dataset cifar10 --arch resnet110  \
--save_path $2 \
--epochs 200 \
--schedule 60 120 160 --use_pretrain 0 --beta 0 \
--gammas  0.2 0.2 0.2 --regularize cutout  \
--learning_rate 0.1 --decay 0.0005 --batch_size 128 \
--rate 0.2  \
--layer_begin 0   --layer_inter 3 --epoch_prune 1
}


pruning 1 ./logs/cifar10_resnet110_rate0.2/


