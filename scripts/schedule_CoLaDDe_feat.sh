#!/bin/bash
# Run from root folder with: bash scripts/schedule.sh

# HK
# DATA_DIR=/home/xi/geovic/LaDDe/train

# TT
DATA_DIR=/geovic/xwang/LaDDe/train_jz/lsun_bedroom/real

REAL_TAG=_E_1000_G_1.0_lsun_bedroom_real

# diff: ladde_feat
python src/train.py \
  experiment=ladde_contrastive_feat \
  trainer.min_epochs=50 \
  trainer.max_epochs=150 \
  data.training_dir=${DATA_DIR} \
  data.real_tag=${REAL_TAG} \
  tags="['feat, CoLaDDe, concat']" \
  +data.preproc_type=concat \
  logger.wandb.name=E_1000_G_1.0_lsun_bedroom_nce_paired \
  task_name=E_1000_G_1.0_lsun_bedroom_nce_paired \
  seed=42 \
  model.triplet_weight=1 \
  model.cls_weight=0 \
  model.optimizer.lr=0.01 \
  data.batch_size=2048 \
  # ckpt_path="/users/xwang/Work/Detection/LaDDe/logs/E_1000_G_1.0_lsun_bedroom_feat_triplet_10x_warmup/runs/2024-01-17_15-46-53/checkpoints/epoch_079.ckpt" \
