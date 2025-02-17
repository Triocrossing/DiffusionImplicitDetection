#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

# train on imagenet, test on imagenet

LOCAL_DIR=data/DIRE/data
TARGET_DIR=lsun_bedroom

# reals
ln -sfn /home/xi/Work/DIRE/dataset/dire/dire/train/lsun_bedroom/adm_jpeg75 ${LOCAL_DIR}/train/${TARGET_DIR}/1_fake
ln -sfn /home/xi/Work/DIRE/dataset/dire/dire/val/lsun_bedroom/adm_jpeg75 ${LOCAL_DIR}/val/${TARGET_DIR}/1_fake

ln -sfn /home/xi/Work/DIRE/dataset/dire/dire/train/lsun_bedroom/real ${LOCAL_DIR}/train/${TARGET_DIR}/0_real
ln -sfn /home/xi/Work/DIRE/dataset/dire/dire/val/lsun_bedroom/real ${LOCAL_DIR}/val/${TARGET_DIR}/0_real


echo "train symlink:"
readlink ${LOCAL_DIR}/train/${TARGET_DIR}/0_real
readlink ${LOCAL_DIR}/val/${TARGET_DIR}/0_real
readlink ${LOCAL_DIR}/train/${TARGET_DIR}/1_fake
readlink ${LOCAL_DIR}/val/${TARGET_DIR}/1_fake

python src/train.py data.training_dir=${LOCAL_DIR}/train/${TARGET_DIR} data.validation_dir=${LOCAL_DIR}/val/${TARGET_DIR} tags="['BCE_1,lsun_bedroom,adm_jpeg75']" task_name=DIRE_${TARGET_DIR}_adm_jpeg75

