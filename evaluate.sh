#!/bin/bash
# Run from root folder with: bash scripts/schedule.sh

# TT
DATA_DIR=$1 # your dataset path / where you should put all the folders in the same directory

REAL_TAG='[pd_real]'

str_list=('[pd_glide]' '[pd_sd4]' '[pd_sd5]' '[pd_wk]' '[pd_biggan]' '[pd_sd4_lr64]' '[pd_sd4_jpeg_30]' '[pd_sd4_s5]')


ckpt=$2

for FAKE_TAG in "${str_list[@]}"; do
    echo "$FAKE_TAG"
cleaned_str="${FAKE_TAG//[\[\]]/}"

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
  logger=csv \
  task_name=${cleaned_str} \
  ckpt_path=${ckpt} \

done
