#!/bin/bash


OUTPATH=/home/sedigheh.eslami/outputs/alignCLIP_openCLIP/shared_transformer_projector_withInModality/logs
BS=512
LR=1e-3
N_EPOCHS=30
MODEL="ViT-L-16"
TRAIN_DATA="/home/sedigheh.eslami/datasets/cc12m/{00000..01242}.tar"
IMAGENET="/home/sedigheh.eslami/datasets/imagenet_1k"
IMAGENETV2="/home/sedigheh.eslami/datasets/cc12m"
#PROJECT_NAME=open_clip_sharedParams_withInModality_CC12M_kisz
PROJECT_NAME=open_clip_sharedParams_withInModality_CC12M_kisz_distributed
ALPHA=1.0
BETA=1.0

wandb login $(cat ~/.wandb_secret)

#python -m training.main --logs=$OUTPATH --save-frequency 2 --zeroshot-frequency 2 --report-to wandb --wandb-project-name=$PROJECT_NAME --train-data=$TRAIN_DATA --warmup 10000  --batch-size=$BS --lr=$LR --wd=0.1 --epochs=$N_EPOCHS --workers=2 --model $MODEL --imagenet-val $IMAGENET --clip-inModality-loss --clip-loss --alpha=$ALPHA --beta=$BETA --nl_semantic_supervision --train-num-samples 10030127 --dataset-type webdataset
torchrun --nproc_per_node 2 -m training.main --logs=$OUTPATH --save-frequency 2 --zeroshot-frequency 2 --report-to wandb --wandb-project-name=$PROJECT_NAME --train-data=$TRAIN_DATA --warmup 10000  --batch-size=$BS --lr=$LR --wd=0.1 --epochs=$N_EPOCHS --workers=2 --model $MODEL --imagenet-val $IMAGENET --clip-inModality-loss --clip-loss --alpha=$ALPHA --beta=$BETA --nl_semantic_supervision --train-num-samples 10030127 --dataset-type webdataset --accum-freq 2 --grad-checkpointing --gather-with-grad
