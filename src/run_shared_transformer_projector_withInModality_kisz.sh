#!/bin/bash


OUTPATH=/raid/sedigheh.eslami/outputs/alignCLIP_openCLIP/shared_transformer_projector_withInModality/logs
BS=512
LR=1e-3
N_EPOCHS=30
#MODEL="ViT-L-16-384-dim"
#MODEL="ViT-L-16"
#MODEL="ViT-B-16"
#MODEL="ViT-XL-16"
#MODEL="ViT-S-16"
MODEL="ViT-M-16"
TRAIN_DATA="/raid/sedigheh.eslami/datasets/cc12m/cc12m/{00000..01242}.tar"
#PROJECT_NAME=open_clip_sharedParams_withInModality_384D_CC12M_kisz
#PROJECT_NAME=open_clip_sharedParams_withInModality_0.2Temperature_CC12M_kisz
#PROJECT_NAME=open_clip_sharedParams_withInModality_0.01Temperature_CC12M_kisz
#PROJECT_NAME=open_clip_sharedParams_withInModality_ViTB_CC12M_kisz
PROJECT_NAME=open_clip_sharedParams_withInModality_CC12M_kisz
WANDBDIR=/raid/sedigheh.eslami/outputs/wandb
##################### CC3M
#OUTPATH=/raid/sedigheh.eslami/outputs/alignCLIP_openCLIP/shared_transformer_projector_withInModality/cc3m/logs
#TRAIN_DATA=/raid/sedigheh.eslami/datasets/cc3m/Train_GCC-training_output.tsv
#PROJECT_NAME=open_clip_sharedParams_withInModality_CC3M_kisz

ALPHA=1
BETA=0.5

wandb login $(cat ~/.wandb_secret)

#python -m training.main --logs=$OUTPATH --save-frequency 2 --report-to wandb --wandb-project-name=$PROJECT_NAME --train-data=$TRAIN_DATA --warmup 10000  --batch-size=$BS --lr=$LR --wd=0.1 --epochs=$N_EPOCHS --workers=2 --model $MODEL --clip-inModality-loss --clip-loss --alpha=$ALPHA --beta=$BETA --nl_semantic_supervision --train-num-samples 10030127 --dataset-type webdataset --separate_text --separate_image --rescale_ablation --wandb-dir $WANDBDIR

python -m training.main --logs=$OUTPATH --save-frequency 2 --report-to wandb --wandb-project-name=$PROJECT_NAME --train-data=$TRAIN_DATA --warmup 10000  --batch-size=$BS --lr=$LR --wd=0.1 --epochs=$N_EPOCHS --workers=2 --model $MODEL --clip-inModality-loss --clip-loss --alpha=$ALPHA --beta=$BETA --nl_semantic_supervision --train-num-samples 10030127 --dataset-type webdataset --separate_image --wandb-dir $WANDBDIR
#python -m training.main --logs=$OUTPATH --save-frequency 2 --report-to wandb --wandb-project-name=$PROJECT_NAME --train-data=$TRAIN_DATA --warmup 10000  --batch-size=$BS --lr=$LR --wd=0.1 --epochs=$N_EPOCHS --workers=2 --model $MODEL --clip-inModality-loss --clip-loss --alpha=$ALPHA --beta=$BETA --nl_semantic_supervision --train-num-samples 10030127 --dataset-type webdataset --separate_image --rescale_ablation --wandb-dir $WANDBDIR

#python -m training.main --logs=$OUTPATH --save-frequency 2 --report-to wandb --wandb-project-name=$PROJECT_NAME --train-data=$TRAIN_DATA --warmup 10000  --batch-size=$BS --lr=$LR --wd=0.1 --epochs=$N_EPOCHS --workers=2 --model $MODEL --clip-inModality-loss --clip-loss --alpha=$ALPHA --beta=$BETA --nl_semantic_supervision --train-num-samples 10030127 --dataset-type webdataset --separate_text --wandb-dir $WANDBDIR

#python -m training.main --logs=$OUTPATH --save-frequency 2 --report-to wandb --wandb-project-name=$PROJECT_NAME --train-data=$TRAIN_DATA --warmup 10000  --batch-size=$BS --lr=$LR --wd=0.1 --epochs=$N_EPOCHS --workers=2 --model $MODEL --clip-inModality-loss --clip-loss --alpha=$ALPHA --beta=$BETA --nl_semantic_supervision --train-num-samples 10030127 --dataset-type webdataset --separate_text --separate_image --wandb-dir $WANDBDIR

#python -m training.main --logs=$OUTPATH --save-frequency 2 --zeroshot-frequency 2 --report-to wandb --wandb-project-name=$PROJECT_NAME --train-data=$TRAIN_DATA --warmup 10000  --batch-size=$BS --lr=$LR --wd=0.1 --epochs=$N_EPOCHS --workers=2 --model $MODEL --clip-inModality-loss --clip-loss --alpha=$ALPHA --beta=$BETA --nl_semantic_supervision --train-num-samples 10030127 --dataset-type webdataset --separate_text --resume "/home/sedigheh.eslami/outputs/alignCLIP_openCLIP/shared_transformer_projector_withInModality/logs/2024_02_26-10_57_28-model_ViT-L-16-lr_0.001-b_512-j_2-p_amp-alpha_0.5-beta_0.5-pairwise_False-alpha_0.5-semantic_1.0/checkpoints/epoch_4.pt"
#python -m training.main --logs=$OUTPATH --save-frequency 2 --zeroshot-frequency 2 --report-to wandb --wandb-project-name=$PROJECT_NAME --train-data=$TRAIN_DATA --warmup 10000  --batch-size=$BS --lr=$LR --wd=0.1 --epochs=$N_EPOCHS --workers=2 --model $MODEL --clip-inModality-loss --clip-loss --alpha=$ALPHA --beta=$BETA --nl_semantic_supervision --train-num-samples 10030127 --dataset-type webdataset --separate_text --separate_image --resume "/home/sedigheh.eslami/outputs/alignCLIP_openCLIP/shared_transformer_projector_withInModality/logs/2024_02_26-10_57_28-model_ViT-L-16-lr_0.001-b_512-j_2-p_amp-alpha_0.5-beta_0.5-pairwise_False-alpha_0.5-semantic_1.0/checkpoints/epoch_4.pt"

#torchrun --nproc_per_node 1 -m training.main --logs=$OUTPATH --save-frequency 2  --report-to wandb --wandb-project-name=$PROJECT_NAME --train-data=$TRAIN_DATA --warmup 10000  --batch-size=$BS --lr=$LR --wd=0.1 --epochs=$N_EPOCHS --workers=2 --model $MODEL --clip-inModality-loss --clip-loss --alpha=$ALPHA --beta=$BETA --nl_semantic_supervision --train-num-samples 10030127 --dataset-type webdataset --accum-freq 2 --grad-checkpointing --gather-with-grad

############## CC3M
#python -m training.main --logs=$OUTPATH --save-frequency 2 --report-to wandb --wandb-project-name=$PROJECT_NAME --train-data=$TRAIN_DATA --warmup 10000  --batch-size=$BS --lr=$LR --wd=0.1 --epochs=$N_EPOCHS --workers=2 --model $MODEL --clip-inModality-loss --clip-loss --alpha=$ALPHA --beta=$BETA --nl_semantic_supervision --separate_image --wandb-dir $WANDBDIR