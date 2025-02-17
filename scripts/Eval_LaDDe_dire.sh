#!/bin/bash
# Run from root folder with: bash scripts/schedule.sh

# HK
# DATA_DIR=/home/xi/geovic/LaDDe/train

# TT
# DATA_DIR=/geovic/xwang/LaDDe/test

DATA_DIR=/geovic/xwang/LaDDe/test/imagenet
REAL_TAG=real
FAKE_TAG=adm

# DATA_DIR=/geovic/xwang/LaDDe/test/imagenet
# REAL_TAG=real
# FAKE_TAG=admjpeg75

# DATA_DIR=/geovic/xwang/LaDDe/test/lsunbedroom
# REAL_TAG=real
# FAKE_TAG=adm

# DATA_DIR=/geovic/xwang/LaDDe/test/lsunbedroom
# REAL_TAG=real
# FAKE_TAG=admjpeg75

# REAL_TAG=_E_1000_G_1.0_lsunbedroom_real_test
# FAKE_TAG=_E_1000_G_1.0_lsunbedroom_adm_test
# FAKE_TAG=_E_1000_G_1.0_lsunbedroom_admjpeg75_test
# FAKE_TAG=_E_1000_G_1.0_imagenet_adm_test
# FAKE_TAG=_E_1000_G_1.0_imagenet_admjpeg75_test

# diff: ladde_feat
python src/eval.py \
  data=ladde \
  data.training_dir=${DATA_DIR} \
  +data.testing_dir=${DATA_DIR} \
  data.real_tag=${REAL_TAG} \
  data.fake_tag=${FAKE_TAG} \
  data.mode=dire \
  model=dire_resnet \
  model.net._target_=src.models.components.resnet.resnet50 \
  model.net.num_classes=1 \
  tags="['feat, LaDDe, concat']" \
  logger.wandb.name=EVAL_E_1000_G_1.0_lsun_bedroom_dire \
  task_name=EVAL_E_1000_G_1.0_lsun_bedroom_dire \
  ckpt_path=/users/xwang/Work/Detection/LaDDe/logs/E_1000_G_1.0_lsun_bedroom_adm75/runs/2024-01-22_17-40-01/checkpoints/epoch_079.ckpt \
  # ckpt_path=/users/xwang/Work/Detection/LaDDe/logs/E_1000_G_1.0_lsun_bedroom/runs/2024-01-16_21-45-45/checkpoints/epoch_085.ckpt \
  # ckpt_path=/users/xwang/Work/Detection/LaDDe/logs/E_1000_G_1.0_lsun_bedroom/runs/2024-01-16_18-38-44/checkpoints/epoch_059.ckpt \