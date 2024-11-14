#!/bin/bash


OUTPATH=/raid/sedigheh.eslami/outputs/alignCLIP_openCLIP/withInModality/logs
BS=512
LR=1e-3
N_EPOCHS=30
MODEL="ViT-L-16-modified"
#MODEL="ViT-L-16-384dim"
#MODEL="ViT-B-16"
TRAIN_DATA="/raid/sedigheh.eslami/datasets/cc12m/cc12m/{00000..01242}.tar"
PROJECT_NAME=open_clip_withInModality_IIonly_CC12M_kisz
#PROJECT_NAME=open_clip_sharedParams_withInModality_CC12M_kisz_distributed
WANDBDIR=/raid/sedigheh.eslami/outputs/wandb
ALPHA=1
BETA=0.25

wandb login $(cat ~/.wandb_secret)

python -m training.main --logs=$OUTPATH --save-frequency 2 --report-to wandb --wandb-project-name=$PROJECT_NAME --train-data=$TRAIN_DATA --warmup 10000  --batch-size=$BS --lr=$LR --wd=0.1 --epochs=$N_EPOCHS --workers=2 --model $MODEL --clip-inModality-loss --clip-loss --alpha=$ALPHA --beta=$BETA --nl_semantic_supervision --train-num-samples 10030127 --dataset-type webdataset
