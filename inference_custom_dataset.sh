#!/bin/bash
# Run from root folder with: bash scripts/schedule.sh

# TT
DATA_DIR=data_preparation/examples # your dataset path / where you should put all the folders in the same directory

FOLDER_TAG='[reals_latent]'


ckpt=$1

python src/inference.py \
  data=cfg \
  data.val_mode=True \
  data.training_dir=$DATA_DIR \
  +data.testing_dir=$DATA_DIR \
  data.real_tag_list=$FOLDER_TAG \
  data.fake_tag_list=$FOLDER_TAG \
  data.preproc_type=concat2uncond \
  data.batch_size=1 \
  model=cfg_resnet \
  model.net._target_=src.models.components.resnet.resnet_50_cfg \
  model.net.num_classes=1 \
  logger=csv \
  task_name=inference \
  ckpt_path=$1