#!/bin/bash
# Run from root folder with: bash scripts/schedule.sh

# HK
# DATA_DIR=/home/xi/geovic/LaDDe/train

# TT
DATA_DIR=/users/xwang/Work/dataset/SD_Detection_val

REAL_TAG='[GenImage_real_50K]'
# Define a list of strings
str_list=('[mj_val]' '[sd4]' '[sd5_val]' '[adm_val]' '[glide_val]' '[wk_val]' '[vq]' '[biggan]')
# str_list=('[sd5_jpeg_val]" "[sd5_val]")
# str_list=("[sd4_lr112]" "[sd4_lr64]" "[sd4_jpeg_65]" "[sd4_jpeg_30]" "[sd4_s3]" "[sd4_s5]")

# Loop through the list
for FAKE_TAG in "${str_list[@]}"; do
    echo "$FAKE_TAG"
cleaned_str="${FAKE_TAG//[\[\]]/}"
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
  model.net._target_=src.models.components.resnet.resnet_50_cfg \
  model.net.num_classes=1 \
  tags="['CFG, eval']" \
  logger.wandb.name=${cleaned_str} \
  task_name=${cleaned_str} \
  ckpt_path=logs/CFG_vs_Noise_SD1.4_concat2uncondb_bs512_0.5_newdata_5_42/runs/2024-03-23_11-34-31/checkpoints/epoch_021.ckpt \
  # ckpt_path=logs/CFG_vs_Noise_SD1.4_post_sweep_42/runs/2024-07-18_01-57-23/checkpoints/last.ckpt \
  # ckpt_path=logs/CFG_vs_Noise_SD1.4_concat2uncondb_bs512_0.5_newdata_5_42/runs/2024-03-23_11-34-31/checkpoints/epoch_021.ckpt \ best

done