#!/bin/bash


TARGET=LATENT_INV_E_1000_G_1.0_imagenet_adm
python pkl_proc.py /geovic/xwang/LaDDe/train_jz/imagenet/$TARGET /geovic/xwang/LaDDe/train_jz/imagenet/TH_$TARGET

TARGET=LATENT_REC_E_1000_G_1.0_imagenet_adm
python pkl_proc.py /geovic/xwang/LaDDe/train_jz/imagenet/$TARGET /geovic/xwang/LaDDe/train_jz/imagenet/TH_$TARGET

# TARGET=LATENT_REC_E_1000_G_1.0_lsun_bedroom_adm
# python pkl_proc.py /geovic/xwang/LaDDe/train_jz/$TARGET /geovic/xwang/LaDDe/train_jz/TH_$TARGET

# TARGET=LATENT_REC_E_1000_G_1.0_lsun_bedroom_real
# python pkl_proc.py /geovic/xwang/LaDDe/train_jz/$TARGET /geovic/xwang/LaDDe/train_jz/TH_$TARGET


# # test
# TARGET=LATENT_INV_E_1000_G_1.0_imagenet_adm_test
# python pkl_proc.py /geovic/xwang/LaDDe/test/$TARGET /geovic/xwang/LaDDe/test/TH_$TARGET

# TARGET=LATENT_INV_E_1000_G_1.0_imagenet_admjpeg75_test
# python pkl_proc.py /geovic/xwang/LaDDe/test/$TARGET /geovic/xwang/LaDDe/test/TH_$TARGET

# TARGET=LATENT_INV_E_1000_G_1.0_imagenet_real_test
# python pkl_proc.py /geovic/xwang/LaDDe/test/$TARGET /geovic/xwang/LaDDe/test/TH_$TARGET

# # bed
# TARGET=LATENT_INV_E_1000_G_1.0_lsunbedroom_adm_test
# python pkl_proc.py /geovic/xwang/LaDDe/test/$TARGET /geovic/xwang/LaDDe/test/TH_$TARGET

# TARGET=LATENT_INV_E_1000_G_1.0_lsunbedroom_admjpeg75_test
# python pkl_proc.py /geovic/xwang/LaDDe/test/$TARGET /geovic/xwang/LaDDe/test/TH_$TARGET

# TARGET=LATENT_INV_E_1000_G_1.0_lsunbedroom_real_test
# python pkl_proc.py /geovic/xwang/LaDDe/test/$TARGET /geovic/xwang/LaDDe/test/TH_$TARGET


# # rec
# TARGET=LATENT_REC_E_1000_G_1.0_imagenet_adm_test
# python pkl_proc.py /geovic/xwang/LaDDe/test/$TARGET /geovic/xwang/LaDDe/test/TH_$TARGET

# TARGET=LATENT_REC_E_1000_G_1.0_imagenet_admjpeg75_test
# python pkl_proc.py /geovic/xwang/LaDDe/test/$TARGET /geovic/xwang/LaDDe/test/TH_$TARGET

# TARGET=LATENT_REC_E_1000_G_1.0_imagenet_real_test
# python pkl_proc.py /geovic/xwang/LaDDe/test/$TARGET /geovic/xwang/LaDDe/test/TH_$TARGET

# TARGET=LATENT_REC_E_1000_G_1.0_lsunbedroom_adm_test
# python pkl_proc.py /geovic/xwang/LaDDe/test/$TARGET /geovic/xwang/LaDDe/test/TH_$TARGET

# TARGET=LATENT_REC_E_1000_G_1.0_lsunbedroom_admjpeg75_test
# python pkl_proc.py /geovic/xwang/LaDDe/test/$TARGET /geovic/xwang/LaDDe/test/TH_$TARGET

# TARGET=LATENT_REC_E_1000_G_1.0_lsunbedroom_real_test
# python pkl_proc.py /geovic/xwang/LaDDe/test/$TARGET /geovic/xwang/LaDDe/test/TH_$TARGET
