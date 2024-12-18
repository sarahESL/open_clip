#!/bin/bash


OUTPATH=/scratch/seslami/alignCLIP_openCLIP/logs
#BS=520
BS=256  # for L
LR=1e-3
N_EPOCHS=30
#MODEL="ViT-B-32"
#MODEL="ViT-M-32"
MODEL="ViT-L-16-modified"
TRAIN_DATA="/scratch/datasets/vision_language/conceptual_captions/Train_GCC-training_output.tsv"
VAL_DATA="/scratch/datasets/vision_language/conceptual_captions/Validation_GCC-1.1.0-Validation_output.tsv"
IMAGENET="/scratch/seslami/imagenet_1k"
PROJECT_NAME=open_clip_CC3M

python -m training.main --logs=$OUTPATH --save-frequency 2 --zeroshot-frequency 10 --report-to wandb --wandb-project-name=$PROJECT_NAME --train-data=$TRAIN_DATA --val-data=$VAL_DATA --warmup 10000  --batch-size=$BS --lr=$LR --wd=0.1 --epochs=$N_EPOCHS --workers=2 --model $MODEL --imagenet-val $IMAGENET
