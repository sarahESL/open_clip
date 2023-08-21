#!/bin/bash


OUTPATH=/scratch4/seslami/alignCLIP_openCLIP/logs
BS=1000
LR=1e-3
N_EPOCHS=50
MODEL="ViT-B-32"
TRAIN_DATA="/scratch1/datasets/vision_language/conceptual_captions/Train_GCC-training_output.tsv"
VAL_DATA="/scratch1/datasets/vision_language/conceptual_captions/Validation_GCC-1.1.0-Validation_output.tsv"
PROJECT_NAME=open_clip_CC3M

python -m training.main --logs=$OUTPATH --save-frequency 10 --zeroshot-frequency 0 --report-to wandb --wandb-project-name=$PROJECT_NAME --train-data=$TRAIN_DATA --val-data=$VAL_DATA --warmup 10000  --batch-size=$BS --lr=$LR --wd=0.1 --epochs=$N_EPOCHS --workers=2 --model $MODEL
