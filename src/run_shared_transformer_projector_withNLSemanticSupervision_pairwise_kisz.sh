#!/bin/bash


OUTPATH=/home/sedigheh.eslami/outputs/alignCLIP_openCLIP/shared_transformer_projector_withNLSemanticSupervision/logs
BS=512
LR=1e-3
N_EPOCHS=30
MODEL="ViT-L-16"
TRAIN_DATA="/home/sedigheh.eslami/datasets/cc12m/{00000..01242}.tar"
PROJECT_NAME=open_clip_sharedParams_withNLSemanticSupervision_CC12M_kisz
ALPHA=1
SEMANTIC_WEIGHT=0.5

python -m training.main --logs=$OUTPATH --save-frequency 2 --report-to wandb --wandb-project-name=$PROJECT_NAME --train-data=$TRAIN_DATA --warmup 10000  --batch-size=$BS --lr=$LR --wd=0.1 --epochs=$N_EPOCHS --workers=2 --model $MODEL --nl_semantic_supervision --alpha $ALPHA --semantic_weight $SEMANTIC_WEIGHT --train-num-samples 10030127 --dataset-type webdataset --semantic_pairwise
