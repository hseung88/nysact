## HuggingFace ImageNet Training 

- model: deit_small_patch16_224 / swin_tiny_patch4_window7_224

```
torchrun --nproc_per_node=4 ./train.py --model deit_small_patch16_224 --sched cosine --epochs 100 \
--opt mac --lr 0.01 --weight-decay 0.005 --damping 0.1 --tinv 100 \
--workers 16 --warmup-epochs 5 --warmup-lr 0.0001 --min-lr 0.0 --batch-size 256 \
--aug-repeats 3 --aa rand-m9-mstd0.5-inc1 --smoothing 0.1 --remode pixel --reprob 0.25 \
--drop 0.0 --drop-path 0.2 --mixup 0.8 --cutmix 1.0 --data-dir /scratch/hs70639/data/imagenet --pin-mem True
```

