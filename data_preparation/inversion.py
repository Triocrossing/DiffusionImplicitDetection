from typing import Optional, Union, Tuple, List, Callable, Dict
from tqdm import tqdm
import torch
import numpy as np
import ptp_utils
from PIL import Image
from constants import (
    _GUIDANCE_SCALE,
    _NUM_DDIM_STEPS,
    _MY_TOKEN,
    _LOW_RESOURCE,
    _MAX_NUM_WORDS,
)
import os
from data.imagenet_helper import class_index_imagenet
from io import BytesIO
from scipy.ndimage import gaussian_filter


def resize_image_with_factor(image, scale_factor=0.5, resampling_filter=Image.BILINEAR):
    """Resize image by a scale factor and then restore to original size."""
    image = Image.fromarray(np.array(image))
    original_width, original_height = image.size
    downsampled_width = int(original_width * scale_factor)
    downsampled_height = int(original_height * scale_factor)
    downsampled_image = image.resize((downsampled_width, downsampled_height), resample=resampling_filter)
    restored_image = downsampled_image.resize((original_width, original_height), resample=resampling_filter)
    restored_image = np.array(restored_image)
    return restored_image
  
def blur_image(image, sigma=1.0):
    """Apply Gaussian blur to an image."""
    image_array = np.array(image)
    if len(image_array.shape) == 3:
        blurred_array = np.zeros_like(image_array)
        for i in range(image_array.shape[2]):
            blurred_array[:, :, i] = gaussian_filter(image_array[:, :, i], sigma=sigma)
    else:
        blurred_array = gaussian_filter(image_array, sigma=sigma)
    blurred_image = Image.fromarray(blurred_array)
    blurred_image = np.array(blurred_image)
    return blurred_image
  
def compress_image_jpeg(image, quality=85):
    """Compress and decompress image using JPEG format."""
    in_memory_file = BytesIO()
    image.save(in_memory_file, format='JPEG', quality=quality)
    in_memory_file.seek(0)
    decompressed_image = Image.open(in_memory_file)
    decompressed_image = np.array(decompressed_image)
    return decompressed_image
  
def preprocess_image(
    image_path, left=0, right=0, top=0, bottom=0, random_crop=False, sbu_caption=False
, args={}):
    """Preprocess image by loading, resizing, and applying transformations."""
    if type(image_path) is str:
        image = np.array(Image.open(image_path))
        if args["jpeg_quality"]!=100:
            image = compress_image_jpeg(Image.fromarray(image), args["jpeg_quality"])
        if args["sigma"]!=0:
            image = blur_image(image, args["sigma"])
        if args["downsample_factor"]!=1:
            image = resize_image_with_factor(image, args["downsample_factor"])
        
        if len(image.shape) == 2:
            print("grayscale image found, convert to rgb")
            print(image_path)
            image = np.repeat(image[:, :, None], 3, axis=2)
        image = image[:, :, :3]
    else:
        image = image_path
    
    h, w, c = image.shape
    if random_crop:
        left = np.random.randint(0, w // 32)
        right = np.random.randint(0, w // 32)
        top = np.random.randint(0, h // 32)
        bottom = np.random.randint(0, h // 32)

    left = min(left, w - 1)
    right = min(right, w - left - 1)
    top = min(top, h - left - 1)
    bottom = min(bottom, h - top - 1)
    image = image[top : h - bottom, left : w - right]
    h, w, c = image.shape
    if h < w:
        offset = (w - h) // 2
        image = image[:, offset : offset + h]
    elif w < h:
        offset = (h - w) // 2
        image = image[offset : offset + w]

    if sbu_caption:
        crop_size = 300
        start_x = w // 2 - crop_size // 2
        start_y = h // 2 - crop_size // 2
        image = image[start_y : start_y + crop_size, start_x : start_x + crop_size]

    image = np.array(Image.fromarray(image).resize((512, 512)))
    return image


class StableDiffusionLatentGenerator:
    def __init__(
        self,
        model,
        ddim_timesteps=_NUM_DDIM_STEPS,
    ):
        self.model = model
        self.tokenizer = self.model.tokenizer
        self.model.scheduler.set_timesteps(ddim_timesteps)
        self.prompt = None
        self.context = None
        self.ddim_timesteps = ddim_timesteps

    def denoise_step(
        self,
        model_output: Union[torch.FloatTensor, np.ndarray],
        timestep: int,
        sample: Union[torch.FloatTensor, np.ndarray],
    ):
        """Perform one step of denoising."""
        prev_timestep = (
            timestep
            - self.scheduler.config.num_train_timesteps
            // self.scheduler.num_inference_steps
        )
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = (
            self.scheduler.alphas_cumprod[prev_timestep]
            if prev_timestep >= 0
            else self.scheduler.final_alpha_cumprod
        )
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (
            sample - beta_prod_t**0.5 * model_output
        ) / alpha_prod_t**0.5
        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
        prev_sample = (
            alpha_prod_t_prev**0.5 * pred_original_sample + pred_sample_direction
        )
        return prev_sample

    def noise_step(
        self,
        model_output: Union[torch.FloatTensor, np.ndarray],
        timestep: int,
        sample: Union[torch.FloatTensor, np.ndarray],
    ):
        """Perform one step of adding noise."""
        timestep, next_timestep = (
            min(
                timestep
                - self.scheduler.config.num_train_timesteps
                // self.scheduler.num_inference_steps,
                999,
            ),
            timestep,
        )
        alpha_prod_t = (
            self.scheduler.alphas_cumprod[timestep]
            if timestep >= 0
            else self.scheduler.final_alpha_cumprod
        )
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (
            sample - beta_prod_t**0.5 * model_output
        ) / alpha_prod_t**0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
        next_sample = (
            alpha_prod_t_next**0.5 * next_original_sample + next_sample_direction
        )
        return next_sample

    def predict_noise(self, latents, t, context):
        """Predict noise for given latents and timestep."""
        noise_pred = self.model.unet(latents, t, encoder_hidden_states=context)[
            "sample"
        ]
        return noise_pred

    def predict_noise_with_guidance(
        self, latents, t, context, guidance_scale=1, verbose=False
    ):
        """Predict noise with classifier-free guidance."""
        latents_input = torch.cat([latents] * 2)
        noise_pred = self.model.unet(latents_input, t, encoder_hidden_states=context)[
            "sample"
        ]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (
            noise_prediction_text - noise_pred_uncond
        )
        return noise_pred

    @torch.no_grad()
    def decode_latents(self, latents, return_type="np"):
        """Convert latents to image."""
        latents = 1 / 0.18215 * latents.detach()
        image = self.model.vae.decode(latents)["sample"]
        image = (image / 2 + 0.5).clamp(0, 1)
        if return_type == "np":
            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
            image = (image * 255).astype(np.uint8)
        return image

    @torch.no_grad()
    def encode_image(self, image, device="cuda"):
        """Convert image to latents."""
        with torch.no_grad():
            if type(image) is Image:
                image = np.array(image)
            if type(image) is torch.Tensor and image.dim() == 4:
                latents = image
            else:
                image = torch.from_numpy(image).float() / 127.5 - 1
                image = image.permute(2, 0, 1).unsqueeze(0).to(device)
                latents = self.model.vae.encode(image)["latent_dist"].mean
                latents = latents * 0.18215
        return latents

    @torch.no_grad()
    def prepare_prompt_embeddings(self, prompts, batch_size=1):
        """Prepare text embeddings for prompts."""
        if type(prompts) is str:
            prompts = [prompts for _ in range(batch_size)]

        assert (
            len(prompts) == batch_size
        ), "Prompts should be of the same length as batch size"

        null_prompts = ["" for _ in range(len(prompts))]
        uncond_input = self.model.tokenizer(
            null_prompts,
            padding="max_length",
            max_length=self.model.tokenizer.model_max_length,
            return_tensors="pt",
        )
        uncond_embeddings = self.model.text_encoder(
            uncond_input.input_ids.to(self.model.device)
        )[0]

        text_input = self.model.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

        text_embeddings = self.model.text_encoder(
            text_input.input_ids.to(self.model.device)
        )[0]

        self.context = torch.cat([uncond_embeddings, text_embeddings])
        self.prompts = prompts

    @property
    def scheduler(self):
        return self.model.scheduler

    def generate_latent_representations(
        self,
        dataloader,
        args={},
    ):
        """Generate latent representations for a batch of images."""
        self.step_ctr = 0
        res_dict = {}
        saving_names = ["e_cond", "e_uncond", "e_opt"]
        
        if args["imagenet_file"]:
                imagenet_helper = class_index_imagenet(args["imagenet_file"])
        for image_file, prompt in tqdm(dataloader):
            base_fname_image = os.path.splitext(os.path.basename(image_file))[0]
            if args["skip_saved"]:
                exist_file = False
                for name in saving_names:
                    saving_dir = os.path.join(args["output_dir"], name)
                    if os.path.exists(os.path.join(saving_dir, f"{base_fname_image}.pt")):
                        exist_file = True
                    else:
                        exist_file = False
                if exist_file:
                    continue
                
            if args["imagenet_file"]:
                ext_prompt = imagenet_helper.auto_filename_parser(base_fname_image)
                prompt = "a photo of "+ext_prompt
            try:
                _res = self.process_image_and_compute_latents(
                    image_file, prompt, offsets=(0, 0, 0, 0), args=args
                )
                if image_file not in res_dict:
                    res_dict[image_file] = {}
                    res_dict["args"] = args
                    res_dict[image_file]["prompt"] = prompt

                dim_to_mean = 1
                for name in saving_names:
                    _epsilon = _res[name].mean(dim=dim_to_mean).cpu()
                    saving_dir = os.path.join(args["output_dir"], name)
                    if not os.path.exists(saving_dir):
                        os.makedirs(saving_dir)
                    torch.save(_epsilon, os.path.join(saving_dir, f"{base_fname_image}.pt"))
            except Exception as e:
              print(f"Error processing {image_file}: {e}")

        return res_dict

    @torch.no_grad()
    def process_image_and_compute_latents(
        self,
        image_path: str,
        prompt: str,
        offsets=(0, 0, 0, 0),
        random_crop=False,
        args={},
    ):
        """Process image and compute latent representations."""
        self.output_image_format_np = True
        self.prepare_prompt_embeddings(prompt, args["batch_size"])
        uncond_embeddings, cond_embeddings = self.context.chunk(2)

        ptp_utils.register_attention_control(self.model, None)
        image_gt = preprocess_image(image_path, *offsets, random_crop, args=args)
            
        latent = self.encode_image(image_gt)
        image_vae = self.decode_latents(
            latent, "np" if self.output_image_format_np else ""
        )

        latent = latent.repeat(args["batch_size"], 1, 1, 1)

        cond_list, uncond_list, noise_list = [], [], []

        latent = latent.clone().detach()
        early_stop_t = args["early_stop_t"]

        for i in tqdm(range(self.ddim_timesteps)):
            t = self.model.scheduler.timesteps[
                len(self.model.scheduler.timesteps) - i - 1
            ]
            if t > early_stop_t:
                break
            cond_list_t, uncond_list_t, noise_list_t = [], [], []
            for sample in range(args["sample_times"]):
                rand_noise = torch.randn_like(latent, device=latent.device)
                noised_latent = self.model.scheduler.add_noise(latent, rand_noise, t)
                with torch.no_grad():
                    noise_pred_cond = self.predict_noise(
                        noised_latent, t, cond_embeddings
                    )
                    noise_pred_uncond = self.predict_noise(
                        noised_latent, t, uncond_embeddings
                    )
                    cond_list_t.append(noise_pred_cond)
                    uncond_list_t.append(noise_pred_uncond)
                    noise_list_t.append(rand_noise)

            cond_list.append(torch.vstack(cond_list_t).cpu())
            uncond_list.append(torch.vstack(uncond_list_t).cpu())
            noise_list.append(torch.vstack(noise_list_t).cpu())

        e_cond = torch.stack(cond_list)
        e_uncond = torch.stack(uncond_list)
        e_opt = torch.stack(noise_list)

        res = {}
        res["img_vae"] = image_vae
        res["e_cond"] = e_cond
        res["e_uncond"] = e_uncond
        res["e_opt"] = e_opt
        return res