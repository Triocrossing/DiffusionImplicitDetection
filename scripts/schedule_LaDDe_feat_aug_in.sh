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
  

DATA_DIR=/geovic/xwang/LaDDe/train_jz/imagenet

REAL_TAG=real
FAKE_TAG=adm

# AUG_REAL_TAG=real
# AUG_FAKE_TAG=adm75

AUG_REAL_TAG=real
AUG_FAKE_TAG=adm_jpeg75

METHOD=addsubcat
TAG=imagenet_aug_75_${METHOD}

# diff: ladde_feat
python src/train.py \
  experiment=ladde_feat \
  trainer.min_epochs=100 \
  trainer.max_epochs=300 \
  data.training_dir=${DATA_DIR} \
  +data.aug_training_dir=${DATA_DIR} \
  data.real_tag=${REAL_TAG} \
  data.fake_tag=${FAKE_TAG} \
  +data.aug_real_tag=${AUG_REAL_TAG} \
  +data.aug_fake_tag=${AUG_FAKE_TAG} \
  tags="['feat, LaDDe, addsubcat']" \
  logger.wandb.name=E_1000_G_1.0_$TAG \
  task_name=E_1000_G_1.0_$TAG \
  seed=42 \
  model.optimizer.lr=0.005 \
  data.batch_size=2048 \
  +data.preproc_type=$METHOD \