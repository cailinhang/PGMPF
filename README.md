# PGMPF

![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg)
# Prior Gradient Mask Guided Pruning-Aware Fine-Tuning

This repository is the PyTorch implementation of [Prior Gradient Mask Guided Pruning-Aware Fine-Tuning](No Link yet) at AAAI2022.


## ImageNet Experiments

Prune pre-trained resnet34 model. `batchsize=768 = 3 * 256` split among `3` GPUs.

```
python pruning_train_gd_prune_bn.py  -a resnet34  \
    --save_dir ./logs/resnet34-rate-0.6 --rate 0.6 --layer_begin 0 --layer_end 105 --layer_inter 3  \
    --use_pretrain --lr 0.02 --epochs 100 --cos 0 -b 768
```


Prune pre-trained resnet50 model. `batchsize=192 = 3 * 64` split among `3` GPUs.

```
python pruning_train_gd_prune_bn.py  -a resnet50  \
    --save_dir ./logs/resnet50-rate-0.6 --rate 0.6 --layer_begin 0 --layer_end 156 --layer_inter 3  \
    --use_pretrain --lr 0.01 --epochs 200 --cos 0 -b 192
```


## How to convert the pruned model into small ones

In accordance with the implementation of [Soft Filter Pruning](https://github.com/he-y/soft-filter-pruning), 

```
sh scripts/get_small.sh
```

can be used to convert the pruned model of res-18/34/50 into small ones. 

The convertion of each model requires case-by-case processing of the Batch Normalization Layers and Downsampling layers.

Note that we had fixed some errors of the original implementation `utils/get_small_model.py` for resnet18/34 in [Soft Filter Pruning](https://github.com/he-y/soft-filter-pruning) caused by the Downsampling layer. 

Besides, in `utils/get_small_model.py`, we provide the code for testing the acutal running time of the small model on GPU/CPU.
