
## Pytorch ImageNet Training

- model: resnet50 / resnet101

```
torchrun --nproc_per_node=4 train.py --data-path /scratch/hs70639/data/imagenet \
--model resnet50 --batch-size 256 --epochs 100 --print-freq 1000 --opt mac \
--lr 0.01 --momentum 0.9 --damping 0.1 --update 100 --weight-decay 0.005 \
--lr-scheduler cosineannealinglr --lr-warmup-epochs 5 --lr-warmup-method linear \
--auto-augment ta_wide --model-ema --random-erase 0.1 --norm-weight-decay 0.0 \
--label-smoothing 0.1 --mixup-alpha 0.2 --cutmix-alpha 1.0 --train-crop-size 176 \
--val-resize-size 232 --ra-sampler --ra-reps 4 \
--output-dir /scratch/hs70639/project/mac/adaact/pytorch_imagenet/outputs/resnet50_mac_100
```
