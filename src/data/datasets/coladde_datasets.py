import os

# python
from PIL import ImageFile, Image

# torch
import torch
from torch.utils.data import Dataset, random_split
import torchvision.datasets as datasets

from src.data.data_utils import load_pickle
import numpy as np

ImageFile.LOAD_TRUNCATED_IMAGES = True

PREFIX_DICT = {
    "coimage": {
        "POS": "IMG_REC",  # IMG_VAE
        "NEG": "IMG_INV",  # IMG_REC
        "EXT": [".png", ".jpg", ".jpeg"],
    },
    "cofeat": {
        "POS": "TH_LATENT_INV",
        "NEG": "TH_LATENT_REC",
        "EXT": [".pkl", ".pth"],
    },
}

# TODO: Sanity Test
class CoLaDDeDataset(Dataset):
    def __init__(
        self,
        root,
        real_tag="0_real",
        dataset_mode="cofeat",
        transform=None,
    ):
        """
        Args:
            root_dir (string): Directory with all the pickle files in different subdirectories.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root
        self.transform = transform
        self.real_tag = real_tag

        assert dataset_mode in ["cofeat", "coimage"]
        self.dataset_mode = dataset_mode

        self.samples = self._load_samples()

    def _load_samples(self):
        # target and negative samples
        _2samples = []

        # target, negative, positive samples
        samples = []
        target_path = os.path.join(self.root_dir, self.real_tag)
        classes = os.listdir(target_path)
        found_classes = []
        for cls in classes:
            # only take real samples
            if PREFIX_DICT[self.dataset_mode]["POS"] in cls:
                found_classes.append(cls)
        
        # should be only one class
        assert len(found_classes) == 1

        for cls in found_classes:
            cls_path_inv = os.path.join(target_path, cls)
            for subroot, subdirs, files in os.walk(cls_path_inv):
                for fname in files:
                    if fname.lower().endswith(
                        tuple(PREFIX_DICT[self.dataset_mode]["EXT"])
                    ):
                        file_path_inv = os.path.join(subroot, fname)
                        file_path_rec = os.path.join(subroot, fname).replace(
                            PREFIX_DICT[self.dataset_mode]["POS"],
                            PREFIX_DICT[self.dataset_mode]["NEG"],
                        )

                        if not os.path.isfile(file_path_inv) or not os.path.isfile(
                            file_path_rec
                        ):
                            print(f"File not found: {file_path_inv} or {file_path_rec}")
                            continue
                        item = (file_path_inv, file_path_rec)
                        _2samples.append(item)

        print(f"Dataset: Found {len(_2samples)} samples")

        # sample random positive
        for sample in _2samples:
            target = sample[0]
            negative = sample[1]
            positive = None
            while positive is None or positive == target:
                positive = _2samples[np.random.randint(0, len(_2samples))][0]
            # random positive

            samples.append((target, negative, positive))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path_inv, file_path_rec, rand_file_path_inv = self.samples[idx]

        # TODO: change this 0 and -1
        if self.dataset_mode == "cofeat":
            # sample_inv = load_pickle(file_path_inv, True)[0].squeeze(0)
            # sample_rec = load_pickle(file_path_rec, True)[-1].squeeze(0)
            # rand_sample_inv = load_pickle(rand_file_path_inv, True)[0].squeeze(0)
            sample_inv = torch.load(file_path_inv)[0].squeeze(0)
            sample_rec = torch.load(file_path_rec)[0].squeeze(0)
            rand_sample_inv = torch.load(rand_file_path_inv)[0].squeeze(0)

        elif self.dataset_mode == "coimage":
            sample_inv = Image.open(file_path_inv)
            sample_rec = Image.open(file_path_rec)
            rand_sample_inv = Image.open(rand_file_path_inv)

        # transform image if needed
        if self.transform and self.dataset_mode == "coimage":
            sample_inv = self.transform(sample_inv)
            sample_rec = self.transform(sample_rec)
            rand_sample_inv = self.transform(rand_sample_inv)

        # target sample, positive sample, negative sample
        return sample_inv, rand_sample_inv, sample_rec
