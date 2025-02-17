#!/bin/bash
# Run from root folder with: bash scripts/schedule.sh

# HK
# DATA_DIR=/home/xi/geovic/LaDDe/train

# TT
DATA_DIR=/geovic/xwang/LaDDe/train_jz/lsun_bedroom

REAL_TAG=real
FAKE_TAG=adm75

python src/train.py \
  experiment=ladde_dire \
  data.training_dir=${DATA_DIR} \
  data.real_tag=${REAL_TAG} \
  data.fake_tag=${FAKE_TAG} \
  tags="['test dataset']" \
  logger.wandb.name=E_1000_G_1.0_lsun_bedroom_adm75 \
  task_name=E_1000_G_1.0_lsun_bedroom_adm75 \
  seed=42 \