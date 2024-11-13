#!/bin/bash


OUTPATH=/raid/sedigheh.eslami/outputs/alignCLIP_openCLIP/logs
BS=512
LR=1e-3
N_EPOCHS=30
#MODEL="ViT-L-16-modified"
#MODEL="ViT-XL-16"
#MODEL="ViT-S-16"
MODEL="ViT-M-16"
TRAIN_DATA="/raid/sedigheh.eslami/datasets/cc12m/cc12m/{00000..01242}.tar"
PROJECT_NAME=open_clip_CC12M_kisz

wandb login $(cat ~/.wandb_secret)

#torchrun --nproc_per_node 2 -m training.main --logs=$OUTPATH --save-frequency 2 --zeroshot-frequency 5 --report-to wandb --wandb-project-name=$PROJECT_NAME --train-data=$TRAIN_DATA --train-num-samples 10030127 --warmup 2000  --batch-size=$BS --lr=$LR --wd=0.1 --epochs=$N_EPOCHS --workers 4 --model $MODEL --imagenet-val $IMAGENET --imagenet-v2 $IMAGENETV2 --precision amp --dataset-type webdataset --accum-freq 2 --grad-checkpointing --gather-with-grad
#python -m training.main --logs=$OUTPATH --save-frequency 2 --zeroshot-frequency 5 --report-to wandb --wandb-project-name=$PROJECT_NAME --train-data=$TRAIN_DATA --train-num-samples 10030127 --warmup 2000  --batch-size=$BS --lr=$LR --wd=0.1 --epochs=$N_EPOCHS --workers 2 --model $MODEL --imagenet-val $IMAGENET --imagenet-v2 $IMAGENETV2 --precision amp --dataset-type webdataset
CUDA_LAUNCH_BLOCKING=1 python -m training.main --logs=$OUTPATH --save-frequency 2 --report-to wandb --wandb-project-name=$PROJECT_NAME --train-data=$TRAIN_DATA --train-num-samples 10030127 --warmup 10000  --batch-size=$BS --lr=$LR --wd=0.1 --epochs=$N_EPOCHS --workers 2 --model $MODEL --precision amp --dataset-type webdataset
