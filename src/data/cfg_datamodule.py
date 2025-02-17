# system
from typing import Any, Dict, Optional, Tuple

# python
from PIL import ImageFile

# torch
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.transforms import transforms
from src.data.data_utils import get_bal_sampler

# datasets
from src.data.datasets.cfg_datasets import CFGFeatureDataset
from lightning import LightningDataModule


ImageFile.LOAD_TRUNCATED_IMAGES = True

# Usage
transform = transforms.Compose(
    [
        # Define your transformations here
    ]
)


###########################################################
class CFGDataModule(LightningDataModule):
    """
    Data module for CFGDataModule dataset, for cfg_vs_noise method
    """

    def __init__(
        self,
        data_dir: str = "data/",
        train_test_split: Tuple[float, float, float] = (0.95, 0.05),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        **kwargs,
    ) -> None:

        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # data are features so no need to resize
        self.transforms = transforms.Compose([])
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size

    def build_dataset(self, data_dir, val_mode=False, data_transforms=None):
        dataset = CFGFeatureDataset(
            root=data_dir,
            real_tag_list=self.hparams.real_tag_list,
            fake_tag_list=self.hparams.fake_tag_list,
            transform=data_transforms,
            preproc_type=self.hparams.preproc_type,
            val_mode=self.hparams.val_mode or val_mode,
        )

        return dataset

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
            self.batch_size_per_device = (
                self.hparams.batch_size // self.trainer.world_size
            )
        if "testing_dir" in self.hparams and self.hparams.testing_dir is not None:
            # load test
            self.data_test = self.build_dataset(
                self.hparams.testing_dir,
            )

        if "validation_dir" in self.hparams and self.hparams.validation_dir is not None:
            # load val
            self.data_val = self.build_dataset(
                self.hparams.validation_dir,
                val_mode=True,
            )

        if "training_dir" in self.hparams and self.hparams.training_dir is not None:
            print("Loading training data")
            # load train and val
            trainset = self.build_dataset(
                self.hparams.training_dir,
            )

            if self.data_test:
                # exist data_test no need to split
                self.data_train = trainset
            elif self.data_val:
                # split train and test
                dataset = ConcatDataset(datasets=[trainset])
                self.data_train, self.data_test = random_split(
                    dataset=dataset,
                    lengths=self.hparams.train_test_split,
                    generator=torch.Generator().manual_seed(
                        42
                    ),  # TODO: not random this?
                )
            else:
                # split train and val and test
                dataset = ConcatDataset(datasets=[trainset])
                self.data_train, self.data_val, self.data_test = random_split(
                    dataset=dataset,
                    lengths=self.hparams.train_test_split,
                    generator=torch.Generator().manual_seed(
                        42
                    ),  # TODO: not random this?
                )
            self.target_list = trainset.targets

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        sampler = (
            get_bal_sampler(self.data_train, self.target_list)
            if self.hparams.class_bal
            else None
        )
        shuffle = (
            not self.hparams.serial_batches
            if (self.hparams.isTrain and not self.hparams.class_bal)
            else False
        )
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
    _ = CFGDataModule()
