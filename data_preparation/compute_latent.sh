OUTPUT_DIR=examples/reals_latent
DIR_TO_IMAGE=examples/reals
IMAGENET_FILE=data/imagenet_class_index.json

# Model parameters
model=sd1.4
ddim_steps=10
num_samples=20
num=10

# Run generation
python gen_latent.py \
    --image_dir=$DIR_TO_IMAGE \
    --num=${num} \
    --output_dir=${OUTPUT_DIR} \
    --ddim_timesteps=$ddim_steps \
    --model=${model} \
    --num_samples=${num_samples} \
    --skip_saved