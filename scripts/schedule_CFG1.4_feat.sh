#!/bin/bash
# Run from root folder with: bash scripts/schedule.sh

# DATA_DIR=/geovic/xwang/DM_Generated
DATA_DIR=/users/xwang/Work/dataset/SD_Detection
DATA_VAL_DIR=/users/xwang/Work/dataset/SD_Detection_val
# DATA_DIR=/home/xi/Work/dataset/SD_Detection

REAL_TAG='[GenImage_real_50K,GenImage_real_50K_suppl_samp100,GenImage_real_50K_suppl_samp20,GenImage_real_50K_suppl_samp20_titan_160k]'
FAKE_TAG='[sd4_suppl_samp20_titan_160k,sd4_suppl_samp20_hk,sd4_suppl_samp20_jz,sd4_suppl_samp20_titan,mj_val,sd4,sd5_val,adm_val,glide_val,wk_val,vq,biggan]' #,sd4_lr112,sd4_jpeg_65]'
# FAKE_TAG='[sd4]' #,sd4_lr112,sd4_jpeg_65]'
# FAKE_TAG='[sd4 GenImage_mj]'

BS=512
seed=43

# diff: ladde_feat
python src/train.py \
  experiment=cfg_feat \
  trainer.min_epochs=10 \
  trainer.max_epochs=250 \
  data.training_dir=${DATA_DIR} \
  +data.validation_dir=${DATA_VAL_DIR} \
  data.real_tag_list=${REAL_TAG} \
  data.fake_tag_list=${FAKE_TAG} \
  tags="['feat']" \
  logger.wandb.name=CFG_vs_Noise_SD1.4_concat2uncondb_bs${BS}_0.5_newdata_5_${seed} \
  task_name=CFG_vs_Noise_SD1.4_concat2uncondb_bs${BS}_0.5_newdata_5_${seed} \
  seed=${seed} \
  model.optimizer.lr=0.01 \
  model.net._target_=src.models.components.resnet.resnet_50_cfg \
  data.batch_size=${BS} \
  data.preproc_type=concat2uncondb \
  logger.wandb.project=CFG_vs_Noise_Detection \