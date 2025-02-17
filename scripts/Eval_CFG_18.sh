#!/bin/bash
# Run from root folder with: bash scripts/schedule.sh

# HK
# DATA_DIR=/home/xi/geovic/LaDDe/train

# TT
DATA_DIR=/users/xwang/Work/dataset/SD_Detection_val

REAL_TAG='[GenImage_real_50K]'
# Define a list of strings
# str_list=("[sd4]" "[GenImage_adm]" "[GenImage_biggan]" "[GenImage_mj]" "[GenImage_sd1p5]" "[GenImage_wukong]")
# str_list=("[sd5_jpeg_val]" "[sd5_val]")
str_list=("[sd4_lr112]" "[sd4_lr64]" "[sd4_jpeg_65]" "[sd4_jpeg_30]" "[sd4_s3]" "[sd4_s5]")

# Loop through the list
for FAKE_TAG in "${str_list[@]}"; do
    echo "$FAKE_TAG"

# diff: ladde_feat
python src/eval.py \
  data=cfg \
  data.val_mode=True \
  data.training_dir=${DATA_DIR} \
  +data.testing_dir=${DATA_DIR} \
  data.real_tag_list=${REAL_TAG} \
  data.fake_tag_list=${FAKE_TAG} \
  data.preproc_type=concat2uncondb \
  data.batch_size=512 \
  model=cfg_resnet \
  model.net._target_=src.models.components.resnet.resnet_18_cfg \
  model.net.num_classes=1 \
  tags="['CFG, eval']" \
  logger.wandb.name=EVAL_CFG_test \
  task_name=EVAL_CFG_test \
  ckpt_path=logs/resnet18_CFG_vs_Noise_SD1.4_concat2uncondb_bs2048_0.5/runs/2024-03-20_20-03-54/checkpoints/last.ckpt \

done