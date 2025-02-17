#!/bin/bash
# Run from root folder with: bash scripts/schedule.sh

# HK
# DATA_DIR=/home/xi/geovic/LaDDe/train

# TT
# DATA_DIR=/home/xi/Work/dataset/lsun_bedromm
DATA_DIR=/users/xwang/Work/dataset/lsun_bedromm

REAL_TAG='[n1000_cfg_vs_noise_stp10_samp_100_lsunb_real]'
# Define a list of strings
# str_list=("[GenImage_adm]" "[GenImage_biggan]" "[GenImage_mj]" "[GenImage_sd1p5]" "[GenImage_wukong]")
# str_list=("[GenImage_sd1p5]")

# n1000_cfg_vs_noise_stp10_samp_100_lsunb_adm     n1000_cfg_vs_noise_stp10_samp_100_lsunb_mj    n1000_cfg_vs_noise_stp10_samp_100_lsunb_sdv1_jpeg75 n1000_cfg_vs_noise_stp10_samp_100_lsunb_adm_75  n1000_cfg_vs_noise_stp10_samp_100_lsunb_real  n1000_cfg_vs_noise_stp10_samp_100_lsunb_sdv2 n1000_cfg_vs_noise_stp10_samp_100_lsunb_ldm     n1000_cfg_vs_noise_stp10_samp_100_lsunb_sdv1  n1000_cfg_vs_noise_stp10_samp_100_lsunb_vqdiffusion
str_list=('[n1000_cfg_vs_noise_stp10_samp_100_lsunb_adm]' '[n1000_cfg_vs_noise_stp10_samp_100_lsunb_mj]' '[n1000_cfg_vs_noise_stp10_samp_100_lsunb_sdv1_jpeg75]' '[n1000_cfg_vs_noise_stp10_samp_100_lsunb_adm_75]' '[n1000_cfg_vs_noise_stp10_samp_100_lsunb_sdv2]' '[n1000_cfg_vs_noise_stp10_samp_100_lsunb_ldm]' '[n1000_cfg_vs_noise_stp10_samp_100_lsunb_sdv1]' '[n1000_cfg_vs_noise_stp10_samp_100_lsunb_vqdiffusion]')

# Loop through the list
for FAKE_TAG in "${str_list[@]}"; do
    echo "$FAKE_TAG"
cleaned_str="${FAKE_TAG//[\[\]]/}"

# diff: ladde_feat
python src/eval.py \
  data=cfg \
  data.training_dir=${DATA_DIR} \
  +data.testing_dir=${DATA_DIR} \
  data.real_tag_list=${REAL_TAG} \
  data.fake_tag_list=${FAKE_TAG} \
  data.preproc_type=subconcat3 \
  data.batch_size=512 \
  model=cfg_resnet \
  model.net._target_=src.models.components.resnet.resnet_50_cfg \
  model.net.num_classes=1 \
  tags="['CFG, eval']" \
  logger.wandb.name=EVAL_CFG_on1.5_${cleaned_str} \
  task_name=EVAL_CFG_DIRE_on1.5_${cleaned_str} \
  ckpt_path=logs/CFG_vs_Noise_SD1.5_subconcat3/runs/2024-03-18_22-09-46/checkpoints/last.ckpt \

done