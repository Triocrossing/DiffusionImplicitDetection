#!/bin/bash

TEST_CKPT=/home/xi/Work/LaDDe/logs/DIRE_imagenet_jpeg75/runs/2024-01-11_01-22-21/checkpoints/epoch_029.ckpt
NAME=DIRE_ImageN_jpeg75_
TAG=last

# echo "training on $TEST_CKPT and test on imagenet png"
# python scripts/tester.py --ckpt $TEST_CKPT --exp_name ${NAME}_${TAG} --test_mode png --test_set imagenet --test_method adm

python scripts/tester.py --ckpt $TEST_CKPT --exp_name ${NAME}_${TAG} --test_mode jpeg75 --test_set imagenet --test_method adm

python scripts/tester.py --ckpt $TEST_CKPT --exp_name ${NAME}_${TAG} --test_mode png --test_set lsun_bedroom --test_method adm

python scripts/tester.py --ckpt $TEST_CKPT --exp_name ${NAME}_${TAG} --test_mode jpeg75 --test_set lsun_bedroom --test_method adm

