#!/bin/bash


OUTPATH=/raid/sedigheh.eslami/outputs/softerCLIP_openCLIP/logs
BS=512
LR=1e-3
N_EPOCHS=30
MODEL="ViT-B-16"
#MODEL="ViT-XL-16"
TRAIN_DATA="/raid/sedigheh.eslami/datasets/cc12m/cc12m/{00000..01242}.tar"
SIM_THETA=0.9
PROJECT_NAME=open_softerclip_CC12M_kisz

wandb login $(cat ~/.wandb_secret)

python -m training.main --logs=$OUTPATH --save-frequency 2 --report-to wandb --wandb-project-name=$PROJECT_NAME --train-data=$TRAIN_DATA --train-num-samples 10030127 --warmup 10000  --batch-size=$BS --lr=$LR --wd=0.1 --epochs=$N_EPOCHS --workers 2 --model $MODEL --precision amp --dataset-type webdataset --soft-loss --nl-semantic-supervision sbert --similarity-threshold $SIM_THETA
