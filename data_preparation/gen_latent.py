# OS
import os
import random

# misc
import json
import fire

# torch related
import torch
from diffusers import (
    StableDiffusionPipeline,
    DDIMScheduler,
)
from torch.utils.data import Dataset, DataLoader

# local
from utils import save_pickle, load_pickle, set_random_seed
from utils_data import (
    get_files_in_dir,
    get_annoation_json,
    find_corresponding_images,
    save_image,
    ith_chunk,
)
from constants import (
    _NUM_DDIM_STEPS,
)

def load_model(model, scheduler, device, ddim_timesteps=1000):
    if model == "sd1.4":
        MY_TOKEN = ""
        sdpipe = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            use_auth_token=MY_TOKEN,
            scheduler=scheduler,
        ).to(device)

        print("done loading")
        try:
            sdpipe.disable_xformers_memory_efficient_attention()
        except AttributeError:
            print("Attribute disable_xformers_memory_efficient_attention() is missing")
        tokenizer = sdpipe.tokenizer

        from inversion import StableDiffusionLatentGenerator

        generator = StableDiffusionLatentGenerator(sdpipe, ddim_timesteps=ddim_timesteps)
    elif model == "sd2":
        MY_TOKEN = ""
        sdpipe = StableDiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2",
            use_auth_token=MY_TOKEN,
            scheduler=scheduler,
        ).to(device)

        print("done loading")
        try:
            sdpipe.disable_xformers_memory_efficient_attention()
        except AttributeError:
            print("Attribute disable_xformers_memory_efficient_attention() is missing")
        tokenizer = sdpipe.tokenizer

        from inversion import StableDiffusionLatentGenerator

        generator = StableDiffusionLatentGenerator(sdpipe, ddim_timesteps=ddim_timesteps)

    return generator

class GODataloader(Dataset):
    def __init__(self, list_of_pairs):
        self.data = list_of_pairs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Retrieve the data at the specified index
        return self.data[idx]

def main(
    image_dir,
    annotation_dir=None,
    num=1000,
    num_epochs=10,
    output_dir=None,
    ddim_timesteps=_NUM_DDIM_STEPS,
    random_crop=False,
    tag="GO",
    model="sd1.5",
    num_samples=10,
    seed=42,
    imagenet_file=None,
    early_stop_t=1000,
    jpeg_quality=100,
    sigma=0,
    downsample_factor=1,
    skip_saved=False,
):
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    world_size = 1
    global_rank = 0

    set_random_seed(seed) # 42 , 18
    is_master = True

    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    print('reading images dir ...')
    img_files = get_files_in_dir(image_dir, num)
    # img_files = ith_chunk(img_files, world_size, global_rank + 1)
    print('reading images done, total images:', len(img_files))

    if annotation_dir is not None:
        annotation_dict = get_annoation_json(annotation_dir)
        pairs = find_corresponding_images(annotation_dict, img_files)
    else: 
        annotation_dict = {}
        pairs = [(img, "a photo of bedroom") for img in img_files]
    
    # prepare DDIM inversion
    scheduler = DDIMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
    )
    
    generator = load_model(model, scheduler, device, ddim_timesteps=ddim_timesteps)
    print(f"Initialize Model: {model}")

    # set the maximum batch size is 50
    batch_size = num_samples
    sample_times = 1
    
    if batch_size > 25:
        sample_times = batch_size // 25
        batch_size = 25
    
    print(f"batch_size: {batch_size}, sample_times: {sample_times}")
    
    additional_args = {
        "random_crop": random_crop,
        "num_epochs": num_epochs,  # show be one
        "sample_times": sample_times,
        "output_dir": output_dir,
        "batch_size": batch_size,
        "early_stop_t": early_stop_t,
        "imagenet_file": imagenet_file if imagenet_file is not None else None,
        "jpeg_quality": jpeg_quality,
        "sigma": sigma,
        "downsample_factor": downsample_factor,
        "skip_saved": skip_saved,
    }
    
    dataloader = GODataloader(pairs)
    if not output_dir:
        print("No output directory specified, not saving results")
        return
     
    res_dict = generator.generate_latent_representations(dataloader, additional_args)

    saved_prompts = {}
    for image_file, res in res_dict.items():
        if image_file == 'args':
            continue
        base_fname_image = os.path.splitext(os.path.basename(image_file))[0]
        saved_prompts[base_fname_image] = res["prompt"]

    # save the prompt in json
    _output_dir_prompt = os.path.join(output_dir, f"prompts_{global_rank}.json")
    json_data = json.dumps(saved_prompts, indent=4)
    with open(_output_dir_prompt, "w") as json_file:
        json_file.write(json_data)

if __name__ == "__main__":
    fire.Fire(main)