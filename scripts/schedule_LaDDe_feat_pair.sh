#!/bin/bash
# Run from root folder with: bash scripts/schedule.sh

# HK
# DATA_DIR=/home/xi/geovic/LaDDe/train

# TT
# DATA_DIR=/geovic/xwang/LaDDe/train_jz/lsun_bedroom

# REAL_TAG=_E_1000_G_1.0_lsun_bedroom_real
# FAKE_TAG=_E_1000_G_1.0_lsun_bedroom_adm

# # diff: ladde_feat
# python src/train.py \
#   experiment=ladde_feat \
#   trainer.max_epochs=500 \
#   data.training_dir=${DATA_DIR} \
#   data.real_tag=${REAL_TAG} \
#   data.fake_tag=${FAKE_TAG} \
#   tags="['feat, LaDDe, concat']" \
#   logger.wandb.name=E_1000_G_1.0_lsun_bedroom_bigger_BS \
#   task_name=E_1000_G_1.0_lsun_bedroom_feat_addsubcat_dropout \
#   seed=42 \
#   model.optimizer.lr=0.005 \
#   data.batch_size=2048 \
#   +data.preproc_type=addsubcat \
  

DATA_DIR=/geovic/xwang/LaDDe/train_jz/lsun_bedroom

REAL_TAG=real
FAKE_TAG=adm

# diff: ladde_feat
python src/train.py \
  experiment=ladde_feat_paired \
  trainer.min_epochs=150 \
  trainer.max_epochs=500 \
  data.training_dir=${DATA_DIR} \
  data.real_tag=${REAL_TAG} \
  data.fake_tag=${FAKE_TAG} \
  tags="['feat, LaDDe, concat']" \
  logger.wandb.name=E_1000_G_1.0_lsun_bedroom_adm_paired \
  task_name=E_1000_G_1.0_lsun_bedroom_adm_paired \
  seed=42 \
  model.optimizer.lr=0.005 \
  data.batch_size=2048 \