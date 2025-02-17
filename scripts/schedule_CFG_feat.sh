#!/bin/bash
# Run from root folder with: bash scripts/schedule.sh

# DATA_DIR=/geovic/xwang/DM_Generated
DATA_DIR=/users/xwang/Work/dataset/SD_Detection
# DATA_DIR=/home/xi/Work/dataset/SD_Detection

REAL_TAG='[GenImage_real_50K_train]'
FAKE_TAG='[GenImage_sd1p5]'

# diff: ladde_feat
python src/train.py \
  experiment=cfg_feat \
  trainer.min_epochs=10 \
  trainer.max_epochs=150 \
  data.training_dir=${DATA_DIR} \
  data.real_tag_list=${REAL_TAG} \
  data.fake_tag_list=${FAKE_TAG} \
  tags="['feat']" \
  logger.wandb.name=CFG_vs_Noise_SD1.5_subconcat3_dim8 \
  task_name=CFG_vs_Noise_SD1.5_subconcat3_dim8 \
  seed=42 \
  model.optimizer.lr=0.01 \
  data.batch_size=128 \
  data.preproc_type=subconcat3 \
  logger.wandb.project=CFG_vs_Noise_Detection \