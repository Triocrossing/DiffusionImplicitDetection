#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

# HK
# DATA_DIR=/home/xi/geovic/LaDDe/train


# DATA_DIR=/geovic/xwang/LaDDe/train
DATA_DIR=/geovic/xwang/LaDDe/train

REAL_TAG=E_1000_G_1.0_lsun_bedroom_real

python src/train.py \
  experiment=ladde_contrastive_image \
  data.training_dir=${DATA_DIR} \
  data.real_tag=${REAL_TAG} \
  data.batch_size=128 \
  model.cls_weight=10 \
  tags="['coimage, first_test']" \
  logger.wandb.name=triplet_first_test \
  task_name=triplet_first_test \

