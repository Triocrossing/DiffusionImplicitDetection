# system
from typing import Any
from random import choice, random
from io import BytesIO
import io

# python
from PIL import Image, ImageFile
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter

# torch
import torch
from torch.utils.data.sampler import WeightedRandomSampler  # for balanced sampling
import pickle


rz_dict = {
    "bilinear": Image.BILINEAR,
    "bicubic": Image.BICUBIC,
    "lanczos": Image.LANCZOS,
    "nearest": Image.NEAREST,
}

# TODO: what ?
ImageFile.LOAD_TRUNCATED_IMAGES = True


# Utils for DIRE dataset
def sample_continuous(s: list):
    if len(s) == 1:
        return s[0]
    if len(s) == 2:
        rg = s[1] - s[0]
        return random() * rg + s[0]
    raise ValueError("Length of iterable s should be 1 or 2.")


def sample_discrete(s: list):
    return s[0] if len(s) == 1 else choice(s)


def gaussian_blur(img: np.ndarray, sigma: float):
    gaussian_filter(img[:, :, 0], output=img[:, :, 0], sigma=sigma)
    gaussian_filter(img[:, :, 1], output=img[:, :, 1], sigma=sigma)
    gaussian_filter(img[:, :, 2], output=img[:, :, 2], sigma=sigma)


def cv2_jpg(img: np.ndarray, compress_val: int) -> np.ndarray:
    img_cv2 = img[:, :, ::-1]
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_val]
    result, encimg = cv2.imencode(".jpg", img_cv2, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg[:, :, ::-1]


def pil_jpg(img: np.ndarray, compress_val: int):
    out = BytesIO()
    img = Image.fromarray(img)
    img.save(out, format="jpeg", quality=compress_val)
    img = Image.open(out)
    # load from memory before ByteIO closes
    img = np.array(img)
    out.close()
    return img


jpeg_dict = {"cv2": cv2_jpg, "pil": pil_jpg}


def jpeg_from_key(img: np.ndarray, compress_val: int, key: str) -> np.ndarray:
    method = jpeg_dict[key]
    return method(img, compress_val)


def get_bal_sampler(dataset, target_list):
    targets = [target_list[i] for i in dataset.indices]

    ratio = np.bincount(targets)
    w = 1.0 / torch.tensor(ratio, dtype=torch.float)
    sample_weights = w[targets]
    return WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights)
    )


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
        else:
            return super().find_class(module, name)


def load_pickle(pickle_path: str, if_cpu_unpickler) -> Any:
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
