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
    "SUBDIR": ["e_opt", "e_cond", "e_uncond"],
    "EXT": [".pt"],
}


def get_roots(root, real_tag=["0_real"], fake_tag=["1_fake"]):
    roots_real = [os.path.join(root, r_tag) for r_tag in real_tag]
    roots_fake = [os.path.join(root, f_tag) for f_tag in fake_tag]
    print("Roots_real: ", roots_real)
    print("Roots_fake: ", roots_fake)

    for r in roots_real:
        if not os.path.isdir(r):
            print("Directory not found: ", r)
            roots_real.remove(r)
    for f in roots_fake:
        if not os.path.isdir(f):
            print("Directory not found: ", f)
            roots_fake.remove(f)
    return roots_real, roots_fake


# we need to load two feats together
class CFGFeatureDataset(Dataset):
    def __init__(
        self,
        root,
        real_tag_list=["GenImage_real_50K"],
        fake_tag_list=["GenImage_sd1p5", "GenImage_mj"],
        preproc_type="none",
        val_mode=False,
        transform=None,
    ):
        super(CFGFeatureDataset, self).__init__()
        print("Dataset: [CFGDataset]")
        print("__preprocess_feat__: type: ", preproc_type)

        self.targets = []
        self.root_dir = root
        self.transform = transform
        self.real_tag_list = real_tag_list
        self.fake_tag_list = fake_tag_list
        self.preproc_type = preproc_type
        self.val_mode = val_mode
        self.samples = self._load_samples()

    def _load_samples_roots(self, roots, idx):
        """
        Load samples from the given roots.

        Args:
          roots (list): List of root directories.
          idx (int): Index of the samples.

        Returns:
          list: List of samples, where each sample is a tuple containing the file paths and the index.
        """
        samples = []
        for root in roots:
            data_path_opt = os.path.join(root, PREFIX_DICT["SUBDIR"][0])
            data_path_cond = os.path.join(root, PREFIX_DICT["SUBDIR"][1])
            data_path_uncond = os.path.join(root, PREFIX_DICT["SUBDIR"][2])

            if not (
                os.path.isdir(data_path_opt)
                and os.path.isdir(data_path_cond)
                and os.path.isdir(data_path_uncond)
            ):
                print(
                    "Data path not found: ",
                    data_path_opt,
                    data_path_cond,
                    data_path_uncond,
                )
                continue

            list_opt = os.listdir(data_path_opt)
            list_cond = os.listdir(data_path_cond)
            list_uncond = os.listdir(data_path_uncond)

            # take common elements in three
            if not (len(list_opt) == len(list_cond) == len(list_uncond)):
                for elem in list_opt:
                    if elem not in list_cond or elem not in list_uncond:
                        list_opt.remove(elem)

            for fname in list_opt:
                file_path_opt = os.path.join(data_path_opt, fname)
                if fname.lower().endswith(tuple(PREFIX_DICT["EXT"])):
                    file_path_cond = os.path.join(data_path_cond, fname)
                    file_path_uncond = os.path.join(data_path_uncond, fname)
                    if (
                        not os.path.isfile(file_path_opt)
                        or not os.path.isfile(file_path_cond)
                        or not os.path.isfile(file_path_uncond)
                    ):
                        print(
                            f"File not found: {file_path_opt} or {file_path_cond} or {file_path_uncond}"
                        )
                        continue

                    item = (file_path_opt, file_path_cond, file_path_uncond, idx)
                    samples.append(item)
                    self.targets.append(idx)
        return samples

    def _load_samples(self):
        """
        Load the samples from the dataset.

        Returns:
          samples (list): A list of samples from the dataset.
        """
        samples = []
        roots_real, roots_fake = get_roots(
            self.root_dir, self.real_tag_list, self.fake_tag_list
        )

        real_samples = self._load_samples_roots(roots_real, 0)
        fake_samples = self._load_samples_roots(roots_fake, 1)
        if self.val_mode:
            real_samples, _ = random_split(
                real_samples, [1000, len(real_samples) - 1000]
            )
            fake_samples, _ = random_split(
                fake_samples, [1000, len(fake_samples) - 1000]
            )
        samples = real_samples + fake_samples

        return samples

    def __len__(self):
        return len(self.samples)

    def __preprocess_feat__(self, sample_opt, sample_cond, sample_uncond):
        if self.preproc_type == "concat3":
            return torch.cat((sample_opt, sample_cond, sample_uncond), dim=1)
        elif self.preproc_type == "concat2cond":
            return torch.cat(
                (sample_cond, sample_cond - sample_uncond, sample_uncond), dim=1
            )
        elif self.preproc_type == "concat2uncond":
            return torch.cat(
                (sample_opt** 2, (sample_opt - sample_uncond) ** 2, sample_uncond** 2), dim=1
            )
        elif self.preproc_type == "concat_opt_uncond":
            return torch.cat((sample_opt, sample_uncond), dim=-1)
        elif self.preproc_type == "concat_opt_cond":
            return torch.cat((sample_opt, sample_cond), dim=-1)
        elif self.preproc_type == "concat_cond_uncond":
            return torch.cat((sample_cond, sample_uncond), dim=-1)
        else:
            print("[self.preproc_type]: ", self.preproc_type)
            print(
                "Preprocessing type not found, must be one of: concat3, concat_opt_uncond, concat_opt_cond, concat_cond_uncond"
            )
            raise NotImplementedError

    def __getitem__(self, idx):
        file_opt, file_cond, file_uncond, label = self.samples[idx]
        sample_opt = torch.load(file_opt)
        sample_cond = torch.load(file_cond)
        sample_uncond = torch.load(file_uncond)

        sample = self.__preprocess_feat__(sample_opt, sample_cond, sample_uncond)

        # save memory
        del sample_opt, sample_cond, sample_uncond

        return sample, label, file_opt
