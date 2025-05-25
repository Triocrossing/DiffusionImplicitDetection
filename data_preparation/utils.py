import pickle
from typing import Any, Tuple
import numpy as np
import torch
import io
import os
from natsort import natsorted
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import random


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
        else:
            return super().find_class(module, name)


def load_pickle(pickle_path: str, if_cpu_unpickler=False):
    """Load a pickle file."""
    with open(pickle_path, "rb") as f:
        if if_cpu_unpickler:
            data = CPU_Unpickler(f).load()
        else:
            data = pickle.load(f)
    return data


def save_pickle(data: Any, pickle_path: str):
    """Save data in a pickle file."""
    with open(pickle_path, "wb") as f:
        pickle.dump(data, f, protocol=4)


def mse_loss(x, y):
    return ((x - y) ** 2).mean()


def l1_loss(x, y):
    return (x - y).abs().mean()


def psnr_loss(img1, img2, max_value=1.0):
    """
    Compute PSNR between two images.

    Parameters:
    - img1, img2: 2D or 3D numpy arrays.

    Returns:
    - PSNR value.
    """
    assert img1.shape == img2.shape, "Input images must have the same dimensions"

    # Compute the mean squared error
    mse = ((img1 - img2) ** 2).mean()

    # Prevent zero division
    if mse == 0:
        return float("inf")

    # Assume 8-bit image, so maximum pixel value is 255.
    # If you're working with different range, adjust accordingly.
    max_pixel_value = max_value

    return 10 * np.log10((max_pixel_value**2) / mse)


def norm(a, b):
    x = np.linalg.norm(abs(a - b), axis=2)
    return np.repeat(x[:, :, np.newaxis], 3, axis=2)


def aggregate_guidances(folder, only_name=None, exclude_name=None):
    """Aggregate guidances from a folder."""
    guidances = []
    for fname in natsorted(os.listdir(folder)):
        if fname.startswith("guidance") and fname.endswith(".pkl"):
            if only_name is not None:
                if only_name not in fname:
                    print(f"Skipping {fname}")
                    continue
            if exclude_name is not None:
                if exclude_name in fname:
                    continue
            _guid = load_pickle(os.path.join(folder, fname), if_cpu_unpickler=True)
            if isinstance(_guid, torch.Tensor):
                _guid = [_guid]
            _guid = torch.stack(_guid)
            guidances.append(_guid)
    guidances = torch.vstack(guidances)
    return guidances


def interpolate_guidance(
    guidance, n_size=1000, filp=True, norm_method="energy_conserv"
):
    if filp:
        intplt_guidance = F.interpolate(
            guidance.flip(0).view(1, 1, -1), size=n_size, mode="linear"
        ).view(-1)
    else:
        intplt_guidance = F.interpolate(
            guidance.view(1, 1, -1), size=n_size, mode="linear"
        ).view(-1)

    if norm_method == "minmax":
        intplt_guidance = min_max_normalize(intplt_guidance)
    elif norm_method == "energy_conserv":
        intplt_guidance = intplt_guidance / intplt_guidance.mean()
    elif norm_method == "none" or "":
        intplt_guidance = intplt_guidance
    else:
        raise ValueError(
            "Invalid norm_method. Choose from 'minmax', 'energy_conserv', 'none'."
        )

    return intplt_guidance


def min_max_normalize(x):
    return (x - x.min()) / (x.max() - x.min())


def normalize(x):
    return x / x.mean()


def aggregate_inpterolate_guidances(
    folder,
    n_size=1000,
    flip=True,
    norm_method="minmax",
    abs=False,
    only_name=None,
    exclude_name=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Aggregate guidances from a folder."""
    res = {}
    guidances = aggregate_guidances(folder, only_name, exclude_name)
    if abs:
        guidances = torch.abs(guidances)
    std_guidance, mean_guidance = torch.std_mean(guidances, dim=0)
    intplt_guidance = interpolate_guidance(mean_guidance, n_size, flip, norm_method)
    res["curve"] = intplt_guidance

    if flip:
        res["std"] = std_guidance.flip(0)
        res["mean"] = mean_guidance.flip(0)
        res["guidances"] = [g.flip(0) for g in guidances]
    else:
        res["std"] = std_guidance
        res["mean"] = mean_guidance
        res["guidances"] = guidances
    return res


def load_image(image_path):
    """Load an image as a PyTorch tensor."""
    image = Image.open(image_path).convert("RGB")  # Ensure image is in RGB format
    transform = transforms.ToTensor()  # Convert to tensor
    return transform(image)


def set_random_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
