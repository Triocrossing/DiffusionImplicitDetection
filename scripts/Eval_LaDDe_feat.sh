#!/bin/bash
# Run from root folder with: bash scripts/schedule.sh

# HK
# DATA_DIR=/home/xi/geovic/LaDDe/train

# TT
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


# diff: ladde_feat
python src/eval.py \
  data=ladde \
  data.training_dir=${DATA_DIR} \
  +data.testing_dir=${DATA_DIR} \
  data.real_tag=${REAL_TAG} \
  data.fake_tag=${FAKE_TAG} \
  data.mode=feat \
  +data.preproc_type=concat \
  model=dire_resnet \
  model.net._target_=src.models.components.resnet.resnet50feat \
  model.net.num_classes=1 \
  tags="['feat, LaDDe, addsubcat']" \
  logger.wandb.name=EVAL_E_1000_G_1.0_lsun_bedroom_feat_addsubcat \
  task_name=EVAL_E_1000_G_1.0_lsun_bedroom_feat_addsubcatt \
  ckpt_path=/users/xwang/Work/Detection/LaDDe/logs/LaDDe_imagenet_aug_concat/2024-01-23_21-08-18/checkpoints/epoch_130.ckpt \
  # ckpt_path=/users/xwang/Work/Detection/LaDDe/logs/E_1000_G_1.0_lsun_bedroom_adm75/runs/2024-01-22_15-46-58/checkpoints/epoch_039.ckpt \
  # ckpt_path=/users/xwang/Work/Detection/LaDDe/logs/E_1000_G_1.0_imagenet_BS/runs/2024-01-19_03-02-19/checkpoints/last.ckpt \
  # ckpt_path=/users/xwang/Work/Detection/LaDDe/logs/E_1000_G_1.0_lsun_bedroom_adm75/runs/2024-01-22_17-40-01/checkpoints/epoch_079.ckpt \
  # ckpt_path=/users/xwang/Work/Detection/LaDDe/logs/E_1000_G_1.0_lsun_bedroom_feat_addsubcat_dropout/runs/2024-01-19_01-49-57/checkpoints/epoch_052.ckpt \
  # ckpt_path=/users/xwang/Work/Detection/LaDDe/logs/E_1000_G_1.0_lsun_bedroom_feat_addsubcat_dropout/runs/2024-01-17_01-15-10/checkpoints/epoch_028.ckpt \

  # ckpt_path=/users/xwang/Work/Detection/LaDDe/logs/E_1000_G_1.0_lsun_bedroom_feat_concat/runs/2024-01-16_22-14-02/checkpoints/epoch_020.ckpt \


  # ckpt_path=/users/xwang/Work/Detection/LaDDe/logs/E_1000_G_1.0_lsun_bedroom_feat_concat/runs/2024-01-16_22-37-41/checkpoints/epoch_019.ckpt \
