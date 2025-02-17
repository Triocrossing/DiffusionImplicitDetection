# system
from typing import Any, Dict, Optional, Tuple
from random import choice, random
from io import BytesIO

# python
from PIL import Image, ImageFile
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter

# torch
import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
import torchvision.transforms.functional as TF
import torchvision.datasets as datasets
from torch.utils.data.sampler import WeightedRandomSampler # for balanced sampling

from src.data.data_utils import rz_dict, sample_continuous, sample_discrete, gaussian_blur, cv2_jpg, pil_jpg, jpeg_dict, jpeg_from_key, get_bal_sampler

# TODO: what ?
ImageFile.LOAD_TRUNCATED_IMAGES = True
    

###########################################################
class DIREDataModule(LightningDataModule):
    """
    Data module for DIRE dataset, aim to replicate and reproduce DIRE code
    """
    def custom_resize(self, img: Image.Image) -> Image.Image:
        interp = sample_discrete(self.hparams.rz_interp)
        return TF.resize(img, self.hparams.loadSize, interpolation=rz_dict[interp])

    def blur_jpg_augment(self, img: Image.Image):
        img: np.ndarray = np.array(img)
        if self.hparams.isTrain:
            if random() < self.hparams.blur_prob:
                sig = sample_continuous(self.hparams.blur_sig)
                gaussian_blur(img, sig)

            if random() < self.hparams.jpg_prob:
                method = sample_discrete(self.hparams.jpg_method)
                qual = sample_discrete(self.hparams.jpg_qual)
                img = jpeg_from_key(img, qual, method)

        return Image.fromarray(img)
      
    def build_dataset(self, data_dir, data_transforms, isTrain):
        dataset = datasets.ImageFolder(root=data_dir, transform=data_transforms)
        return dataset
  
    def data_transforms(self):
        identity_transform = transforms.Lambda(lambda img: img)

        if self.hparams.isTrain or self.hparams.aug_resize:
            rz_func = transforms.Lambda(lambda img: self.custom_resize(img))
        else:
            rz_func = identity_transform

        if self.hparams.isTrain:
            crop_func = transforms.RandomCrop(self.hparams.cropSize)
        else:
            crop_func = transforms.CenterCrop(self.hparams.cropSize) if self.hparams.aug_crop else identity_transform

        if self.hparams.isTrain and self.hparams.aug_flip:
            flip_func = transforms.RandomHorizontalFlip()
        else:
            flip_func = identity_transform

        return transforms.Compose(
                [
                    rz_func,
                    transforms.Lambda(lambda img: self.blur_jpg_augment(img)),
                    crop_func,
                    flip_func,
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    if self.hparams.aug_norm
                    else identity_transform,
                ]
            )

    def __init__(
        self,
        data_dir: str = "data/",
        train_test_split: Tuple[float, float, float] = (0.95, 0.05),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        **kwargs,
    ) -> None:
        """Initialize a `DIREDataModule`.

        :param data_dir: The data directory. Defaults to `"data/"`.
        :param train_val_test_split: The train, validation and test split. Defaults to `(55_000, 5_000, 10_000)`.
        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # data transformations
        self.transforms = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size

    @property
    def num_classes(self) -> int:
        """Get the number of classes.

        :return: The number of MNIST classes (10).
        """
        return 10

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        if "testing_dir" in self.hparams and self.hparams.testing_dir is not None:
            # load test
            self.data_test = self.build_dataset(self.hparams.testing_dir, self.data_transforms(), False)
        
        if "validation_dir" in self.hparams and self.hparams.validation_dir is not None:
            # load val
            self.data_val = self.build_dataset(self.hparams.validation_dir, self.data_transforms(), self.hparams.isTrain)
            
        if "training_dir" in self.hparams and self.hparams.training_dir is not None:
            # load train and val
            trainset = self.build_dataset(self.hparams.training_dir, self.data_transforms(), self.hparams.isTrain)
            
            if self.data_test:
                # existi data_test no need to split
                self.data_train = trainset
            elif self.data_val: 
                # split train and test
                dataset = ConcatDataset(datasets=[trainset])
                self.data_train, self.data_test = random_split(
                    dataset=dataset,
                    lengths=self.hparams.train_test_split,
                    generator=torch.Generator().manual_seed(42), #TODO: not random this?
                )
            else:
                # split train and val and test
                dataset = ConcatDataset(datasets=[trainset])
                self.data_train, self.data_val, self.data_test = random_split(
                    dataset=dataset,
                    lengths=self.hparams.train_test_split,
                    generator=torch.Generator().manual_seed(42), #TODO: not random this?
                )
            

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        sampler = get_bal_sampler(self.data_train) if self.hparams.class_bal else None
        shuffle = not self.hparams.serial_batches if (self.hparams.isTrain and not self.hparams.class_bal) else False
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=shuffle,
            sampler=sampler,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass


if __name__ == "__main__":
    _ = DIREDataModule()
