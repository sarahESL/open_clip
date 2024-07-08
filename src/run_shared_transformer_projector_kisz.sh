#!/bin/bash


OUTPATH=/home/sedigheh.eslami/outputs/alignCLIP_openCLIP/shared_transformer_projector/logs
BS=512
LR=1e-3
N_EPOCHS=30
#MODEL="ViT-L-16-384-dim"
#MODEL="ViT-L-16"
MODEL="ViT-XL-16"
#MODEL="ViT-B-16"
TRAIN_DATA="/raid/sedigheh.eslami/datasets/cc12m/cc12m/{00000..01242}.tar"
#PROJECT_NAME=open_clip_sharedParams_CC12M_kisz
#PROJECT_NAME=open_clip_sharedParams_384Dim_CC12M_kisz
#PROJECT_NAME=open_clip_sharedParams_0.2Temperature_CC12M_kisz
#PROJECT_NAME=open_clip_sharedParams_0.01Temperature_CC12M_kisz
PROJECT_NAME=open_clip_sharedParams_CC12M_kisz

wandb login $(cat ~/.wandb_secret)

python -m training.main --logs=$OUTPATH --save-frequency 2 --report-to wandb --wandb-project-name=$PROJECT_NAME --train-data=$TRAIN_DATA --train-num-samples 10030127 --warmup 10000  --batch-size=$BS --lr=$LR --wd=0.1 --epochs=$N_EPOCHS --workers=2 --model $MODEL --precision amp --dataset-type webdataset --resume "/home/sedigheh.eslami/outputs/alignCLIP_openCLIP/shared_transformer_projector/logs/2024_06_07-12_59_49-model_ViT-XL-16-lr_0.001-b_512-j_2-p_amp/checkpoints/epoch_28.pt"
#python -m training.main --logs=$OUTPATH --save-frequency 2 --report-to wandb --wandb-project-name=$PROJECT_NAME --train-data=$TRAIN_DATA --train-num-samples 10030127 --warmup 10000  --batch-size=$BS --lr=$LR --wd=0.1 --epochs=$N_EPOCHS --workers=2 --model $MODEL --precision amp --dataset-type webdataset --resume "/home/sedigheh.eslami/outputs/alignCLIP_openCLIP/shared_transformer_projector/logs/2024_05_14-16_50_34-model_ViT-L-16-lr_0.001-b_512-j_2-p_amp/checkpoints/epoch_20.pt"
