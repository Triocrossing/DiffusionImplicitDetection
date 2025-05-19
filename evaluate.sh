#!/bin/bash
# Run from root folder with: bash scripts/schedule.sh

# TT
DATA_DIR= # your dataset path / where you should put all the folders in the same directory

REAL_TAG='[pd_real]'

str_list=('[pd_glide_val]' '[pd_sd4_val]' '[pd_sd5_val]' '[pd_wk_val]' '[pd_biggan_val]' '[pd_sd4_lr64]' '[pd_sd4_jpeg_30]' '[pd_sd4_s5]')


ckpt=$1

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
  data.preproc_type=concat2uncond \
  data.batch_size=512 \
  model=cfg_resnet \
  model.net._target_=src.models.components.resnet.resnet_50_cfg \
  model.net.num_classes=1 \
  tags="['CFG, eval']" \
  logger.wandb.name=${cleaned_str} \
  task_name=${cleaned_str} \
  ckpt_path=${ckpt} \

done