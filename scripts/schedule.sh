#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

# train on imagenet, test on imagenet

LOCAL_DIR=data/DIRE/data

# reals
ln -sfn /home/xi/Work/DIRE/dataset/dire/dire/train/imagenet/real ${LOCAL_DIR}/train/imagenet/0_real
ln -sfn /home/xi/Work/DIRE/dataset/dire/dire/val/imagenet/real ${LOCAL_DIR}/val/imagenet/0_real

ln -sfn /home/xi/Work/DIRE/dataset/dire/dire/train/imagenet/adm ${LOCAL_DIR}/train/imagenet/1_fake
ln -sfn /home/xi/Work/DIRE/dataset/dire/dire/val/imagenet/adm ${LOCAL_DIR}/val/imagenet/1_fake

echo "train symlink:"
readlink ${LOCAL_DIR}/train/imagenet/0_real
readlink ${LOCAL_DIR}/val/imagenet/0_real
readlink ${LOCAL_DIR}/train/imagenet/1_fake
readlink ${LOCAL_DIR}/val/imagenet/1_fake

python src/train.py data.training_dir=${LOCAL_DIR}/train/imagenet data.validation_dir=${LOCAL_DIR}/val/imagenet tags="['BCE_1,imagenet,adm_png']" task_name=DIRE_imagenet_adm_png

