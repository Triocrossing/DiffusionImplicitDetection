#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

# train on imagenet, test on imagenet

# HK
# DATA_DIR=/home/xi/geovic/LaDDe/train

# TT
DATA_DIR=/geovic/xwang/LaDDe/train

FOLDER_NAME_REAL=IMG_DIRE_E_501_G_1.0_lsun_bedroom_real
FOLDER_NAME_FAKE=IMG_DIRE_E_501_G_1.0_lsun_bedroom_adm_jpeg75

LOCAL_DIR=data/LaDDe/data

# reals
ln -sfn ${DATA_DIR}/${FOLDER_NAME_REAL} ${LOCAL_DIR}/train/lsun_bedroom/0_real
ln -sfn ${DATA_DIR}/${FOLDER_NAME_FAKE} ${LOCAL_DIR}/train/lsun_bedroom/1_fake

echo "train symlink:"
readlink ${LOCAL_DIR}/train/lsun_bedroom/0_real
readlink ${LOCAL_DIR}/val/lsun_bedroom/0_real
readlink ${LOCAL_DIR}/train/lsun_bedroom/1_fake
readlink ${LOCAL_DIR}/val/lsun_bedroom/1_fake

python src/train.py experiment=ladde data.training_dir=${LOCAL_DIR}/train/lsun_bedroom tags="['e501,g1,lsun_bedroom, LaDDe']" logger.wandb.name=LaDDe_lsun_bedroom_e500_g1.0_jpeg75 task_name=LaDDe_lsun_bedroom_e500_g1.0_jpeg75