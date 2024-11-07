# NysAct: A Scalable Preconditioning Method Using Nystrom Approximation

This repository contains the implementation of **NysAct**, a novel optimization algorithm for deep learning that 
leverages the Nystrom approximation to stabilize and improve the training of large-scale neural networks. 
The algorithm, introduced in our paper accepted at IEEE BigData 2024, provides an efficient alternative to conventional 
second-order optimization methods by approximating the activation covariance matrix.

## Overview

Deep learning models rely heavily on efficient and effective optimization strategies. While first-order methods are 
computationally efficient, they often struggle with slow convergence and generalization. 
Second-order methods, in contrast, can address these limitations but are usually costly in terms of time and memory. 
**NysAct** bridges this gap by providing a scalable first-order optimization method that uses the Nystrom approximation 
for gradient preconditioning. This approach offers:
- **Improved convergence rates** with lower computational costs.
- **Enhanced generalization** by balancing between first-order and second-order optimization strengths.
- **Scalable implementation** suitable for large-scale datasets and models.

## Key Features

- **Nystrom Approximation**: Uses eigenvalue-shifted Nystrom method to approximate the activation covariance matrix efficiently, maintaining accuracy while reducing computational demands.
- **Adaptable to Large Models**: Reduces memory and time complexity, making it suitable for large architectures like ResNet and Vision Transformers.
- **Empirical Performance**: Outperforms traditional first-order and even some second-order methods in accuracy and training time on benchmarks like CIFAR and ImageNet.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/hseung88/nysact.git
cd nysact
```

2. Create a virtual environment:
```bash
conda env create -n nysact -f environments.yml
conda activate nysact
```

## Usage

To train a model using NysAct on CIFAR:
```bash
python main.py --model resnet32 --optim nysact --lr 0.1 --momentum 0.9 --stat_decay 0.95 --damping 1.0 \
--tcov 5 --tinv 50 --weight_decay 0.0005 --rank 10 --sketch gaussian --epoch 200 --run 0;
```

For training on ImageNet:
1. ResNet-50
```bash
cd ./pytorch_imagenet
torchrun --nproc_per_node=4 ./train.py --data-path /data/imagenet --model resnet50 --batch-size 256 --print-freq 1000 \
--opt nysact --lr 0.5 --damping 1.0 --rank_size 50 --lr-scheduler cosineannealinglr --lr-warmup-epochs 5 \
--lr-warmup-method linear --auto-augment ta_wide --epochs 100 --model-ema --random-erase 0.1 --weight-decay 0.00002 \
--norm-weight-decay 0.0 --label-smoothing 0.1 --mixup-alpha 0.2 --cutmix-alpha 1.0 --train-crop-size 176 \
--val-resize-size 232 --ra-sampler --ra-reps 4 --output-dir ./outputs/resnet50_nysact_100
```
2. DeiT-Small
```bash
cd ./hf_imagenet
torchrun --nproc_per_node=4 ./train.py --model deit_small_patch16_224 --sched cosine --epochs 100 --opt nysact \
--lr 0.5 --weight-decay 0.0001 --workers 16 --warmup-epochs 5 --warmup-lr 0.005 --min-lr 0.0 --batch-size 256 \
--aug-repeats 3 --aa rand-m7-mstd0.5-inc1 --smoothing 0.1 --remode pixel --reprob 0.25 \
--drop 0.0 --drop-path 0.2 --mixup 0.8 --cutmix 1.0 --data-dir /data/imagenet --pin-mem True;
```

## Citation
```bash
@inproceedings{seung2024nysact,
  title={NysAct: A Scalable Preconditioned Gradient Descent using Nystrom Approximation},
  author={Seung, Hyunseok and Lee, Jaewoo and Ko, Hyunsuk},
  booktitle={IEEE International Conference on Big Data},
  year={2024}
}
```