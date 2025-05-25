import os
import random
import json
import torch
from PIL import Image
import numpy as np


def get_files_in_dir(directory, num=None):
    list_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            if (
                os.path.isfile(file_path)
                and not file.startswith(".")  # not hidden
                and file.lower().endswith((".png", ".jpg", ".jpeg"))  # image
            ):
                list_files.append(file_path)
    if num:
        return random.sample(list_files, num) if num < len(list_files) else list_files
    return list_files


def sample_files(files, n):
    """Randomly sample N files from a list."""
    if n > len(files):
        print("Warning: Requested more files than are available. Returning all files.")
        return files
    return random.sample(files, n)


def get_annoation_json(annotation_dir):
    # read json
    with open(annotation_dir, "r") as file:
        data = json.load(file)
    return data


def find_corresponding_images(
    annotation_dict,
    image_files,
):
    """Find corresponding image files for a list of annotation files."""
    pairs = []
    for image_dir in image_files:
        base_fname_image = os.path.splitext(os.path.basename(image_dir))[0]
        if base_fname_image in annotation_dict.keys():
            pairs.append((image_dir, annotation_dict[base_fname_image]))
        else:
            print(f"Warning: No image found for annotation {base_fname_image}")
    return pairs


def save_image(image_array, output_dir, subdir, image_file, mode="ow"):
    if image_array is None:
        return
    base_name = os.path.basename(image_file).replace(".jpg", "")
    if not os.path.exists(os.path.join(output_dir, subdir)):
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)
    output_path = os.path.join(output_dir, subdir, f"{base_name}.png")
    if isinstance(image_array, torch.Tensor):
        image_array = image_array.cpu().numpy()
    if isinstance(image_array, np.ndarray):
        image = Image.fromarray(image_array)
    if isinstance(image_array, Image.Image):
        image = image_array
    if os.path.exists(output_path):
        if mode == "ow":
            image.save(output_path)
            return
        elif mode == "keep":
            ctr = 1
            while os.path.exists(output_path):
                ctr += 1
                output_path = os.path.join(output_dir, subdir, f"{base_name}_{ctr}.png")
            image.save(output_path)
            return
    else:
        image.save(output_path)

def ith_chunk(input, n, i):
    """Returns the i-th piece of the dictionary/list split into n parts."""
    if n <= 0:
        raise ValueError("Number of parts must be positive")
    if i < 1 or i > n:
        raise ValueError("i must be between 1 and n")

    if isinstance(input, dict):
        # Convert dictionary to a list of items for easy slicing
        items = list(input.items())
    else:
        items = input
    # Calculate the minimum chunk size
    min_chunk_size = len(items) // n
    # Calculate how many chunks need an extra item
    extra = len(items) % n

    # Find the start and end indices for the i-th chunk
    start = sum(min_chunk_size + (j < extra) for j in range(i - 1))
    end = start + min_chunk_size + (i - 1 < extra)

    # Return the i-th chunk as a dictionary
    if isinstance(input, dict):
        return dict(items[start:end])
    else:
        return items[start:end]  # list