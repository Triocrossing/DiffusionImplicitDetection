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
    "image": {
        "PREF": "IMG_DIRE",
        "EXT": [".png", ".jpg", ".jpeg"],
    },
    "feat": {
        "POS": "TH_LATENT_INV",
        "NEG": "TH_LATENT_REC",
        "EXT": [".pkl", ".pth"],
    },
}


def get_classes(root, real_tag="0_real", fake_tag="1_fake"):
    classes = []
    root_real = os.path.join(root, real_tag)
    root_fake = os.path.join(root, fake_tag)
    assert os.path.isdir(root_real)
    assert os.path.isdir(root_fake)
    classes = os.listdir(root_real)
    classes.extend(os.listdir(root_fake))
    return root_real, root_fake, classes


# we need to load two feats together
class LaDDeFeatureDataset(Dataset):
    def __init__(
        self,
        root,
        real_tag="0_real",
        fake_tag="1_fake",
        preproc_type="concat",
        transform=None,
        aug_real_tag=None,
        aug_fake_tag=None,
    ):
        super(LaDDeFeatureDataset, self).__init__()
        print("Dataset: [LaDDeFeatureDataset]")
        print("__preprocess_feat__: type: ", preproc_type)

        self.root_dir = root
        self.rec_feat_dir = root
        self.transform = transform
        self.real_tag = real_tag
        self.fake_tag = fake_tag
        self.aug_real_tag = aug_real_tag
        self.preproc_type = preproc_type

        self.tags = {"real": self.real_tag, "fake": self.fake_tag}
        if self.aug_real_tag is not None:
            self.tags["aug_real"] = self.aug_real_tag
        if self.aug_fake_tag is not None:
            self.tags["aug_fake"] = self.aug_fake_tag

        self.samples = self._load_samples()

    def _load_samples(self):
        samples = []
        root_real, root_fake, classes = get_classes(
            self.root_dir, self.real_tag, self.fake_tag
        )

        class_to_idx = {}

        for cls in classes:
            if PREFIX_DICT["feat"]["POS"] not in cls:
                continue
            if self.real_tag not in cls and self.fake_tag not in cls:
                continue
            class_to_idx[cls] = (
                0 if self.real_tag in cls else 1 if self.fake_tag in cls else -1
            )

        assert len(class_to_idx) == 2
        assert -1 not in class_to_idx.values()

        # only INVs -> RECs is automatically parsed

        for cls in class_to_idx.keys():
            if self.real_tag in cls:
                cls_path_inv = os.path.join(root_real, cls)
            else:
                cls_path_inv = os.path.join(root_fake, cls)

            # cls_path_inv = os.path.join(self.root_dir, cls)

            for subroot, subdirs, files in os.walk(cls_path_inv):
                for fname in files:
                    if fname.lower().endswith(tuple(PREFIX_DICT["feat"]["EXT"])):
                        file_path_inv = os.path.join(subroot, fname)
                        file_path_rec = os.path.join(subroot, fname).replace(
                            PREFIX_DICT["feat"]["POS"], PREFIX_DICT["feat"]["NEG"]
                        )
                        if not os.path.isfile(file_path_inv) or not os.path.isfile(
                            file_path_rec
                        ):
                            print(f"File not found: {file_path_inv} or {file_path_rec}")
                            continue

                        item = (file_path_inv, file_path_rec, class_to_idx[cls])
                        samples.append(item)

        return samples

    def __len__(self):
        return len(self.samples)

    def __preprocess_feat__(self, sample_inv, sample_rec):
        if self.preproc_type == "concat":
            return torch.cat((sample_inv, sample_rec), dim=-1)
        if self.preproc_type == "addsubcat":
            return torch.cat((sample_inv + sample_rec, sample_inv - sample_rec), dim=-1)

    def __getitem__(self, idx):
        file_path_inv, file_path_rec, label = self.samples[idx]

        # TODO: change this 0 and -1
        sample_inv = torch.load(file_path_inv)[0].squeeze(0)
        sample_rec = torch.load(file_path_rec)[-1].squeeze(0)

        sample = self.__preprocess_feat__(sample_inv, sample_rec)

        del sample_inv, sample_rec

        return sample, label


class LaDDeImageDataset(Dataset):
    def __init__(
        self,
        root,
        real_tag="0_real",
        fake_tag="1_fake",
        transform=None,
    ):
        super(LaDDeImageDataset, self).__init__()
        self.root_dir = root
        self.rec_feat_dir = root
        self.transform = transform
        self.real_tag = real_tag
        self.fake_tag = fake_tag
        self.samples = self._load_samples()

    def _load_samples(self):
        samples = []
        root_real, root_fake, classes = get_classes(
            self.root_dir, self.real_tag, self.fake_tag
        )
        class_to_idx = {}

        for cls in classes:
            if PREFIX_DICT["image"]["PREF"] not in cls:
                continue
            if self.real_tag not in cls and self.fake_tag not in cls:
                continue
            class_to_idx[cls] = (
                0 if self.real_tag in cls else 1 if self.fake_tag in cls else -1
            )

        assert len(class_to_idx) == 2
        assert -1 not in class_to_idx.values()

        # two DIRE folders

        for cls in class_to_idx.keys():
            if self.real_tag in cls:
                cls_path_img = os.path.join(root_real, cls)
            else:
                cls_path_img = os.path.join(root_fake, cls)

            for subroot, subdirs, files in os.walk(cls_path_img):
                for fname in files:
                    if fname.lower().endswith(tuple(PREFIX_DICT["image"]["EXT"])):
                        file_path = os.path.join(subroot, fname)

                        if not os.path.isfile(file_path):
                            print(f"File not found: {file_path}")
                            continue

                        item = (file_path, class_to_idx[cls])
                        samples.append(item)
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        sample_raw = Image.open(file_path)
        if self.transform:
            sample = self.transform(sample_raw)

        return sample, label


# save here to compare the refactoring
class _LaDDeImageDataset(Dataset):
    def __init__(
        self,
        root,
        real_tag="0_real",
        fake_tag="1_fake",
        transform=None,
    ):
        super(_LaDDeImageDataset, self).__init__()
        self.data = datasets.ImageFolder(root=root, transform=transform)
        print("Deprecated Dataset: [_LaDDeImageDataset] \n only for sanity test")
        print(self.data.classes)
        print(self.data.class_to_idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data.__getitem__(idx)
