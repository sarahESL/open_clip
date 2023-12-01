#!/bin/bash


OUTPATH=/scratch/seslami/alignCLIP_openCLIP/shared_transformer_projector_withInModality/logs
BS=256
LR=1e-3
N_EPOCHS=60
MODEL="ViT-L-16"
TRAIN_DATA="/scratch/datasets/vision_language/conceptual_captions/Train_GCC-training_output.tsv"
VAL_DATA="/scratch/datasets/vision_language/conceptual_captions/Validation_GCC-1.1.0-Validation_output.tsv"
IMAGENET="/scratch/seslami/imagenet_1k"
#PROJECT_NAME=open_clip_sharedParams_withInModality_CC3M
PROJECT_NAME=open_clip_adaptive_sharedParams_withInModality_CC3M
#ALPHA=1.0
#BETA=0.1

# python -m training.main --logs=$OUTPATH --save-frequency 2 --zeroshot-frequency 10 --report-to wandb --wandb-project-name=$PROJECT_NAME --train-data=$TRAIN_DATA --val-data=$VAL_DATA --warmup 10000  --batch-size=$BS --lr=$LR --wd=0.1 --epochs=$N_EPOCHS --workers=2 --model $MODEL --imagenet-val $IMAGENET --clip-inModality-loss --clip-loss --alpha=$ALPHA --beta=$BETA
python -m training.main --logs=$OUTPATH --save-frequency 2 --zeroshot-frequency 10 --report-to wandb --wandb-project-name=$PROJECT_NAME --train-data=$TRAIN_DATA --val-data=$VAL_DATA --warmup 10000  --batch-size=$BS --lr=$LR --wd=0.1 --epochs=$N_EPOCHS --workers=2 --model $MODEL --imagenet-val $IMAGENET --clip-inModality-loss --clip-loss --adaptive
