from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy

# from torchmetrics.classification.AveragePrecision import AveragePrecision
from src.utils.simCLR_utils import naive_contrastive_loss
from src.utils.SupConLoss import SupConLoss
import torch.nn.functional as F

from src.utils.viz_utils import visualize_features, stat_feature

from info_nce import InfoNCE

import wandb


class CoLaDDeModule(LightningModule):
    """Example of a `LightningModule` for CoLaDDe classification.

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        **kwargs,
    ) -> None:
        """Initialize a `CoLaDDeModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net

        # loss function
        # TODO: adjust margin and p as needed
        self.criterion_triplet = torch.nn.TripletMarginLoss(margin=1.0, p=2, eps=1e-7)
        self.criterion_cls = torch.nn.BCEWithLogitsLoss()
        self.criterion_simCLR = torch.nn.CrossEntropyLoss()
        self.criterion_infoNCE = InfoNCE(negative_mode="paired")

        # metric objects for calculating and averaging accuracy across batches
        self.train_acc = Accuracy(task="binary", num_classes=2)
        self.val_acc = Accuracy(task="binary", num_classes=2)
        self.test_acc = Accuracy(task="binary", num_classes=2)

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        return self.net(anchor, positive, negative)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        anchor, positive, negative = batch

        (
            feat_anchor,  # positve to anchor
            feat_positive,
            feat_negative,
            anchor_output,
            positive_output,
            negative_output,
        ) = self.forward(
            anchor=anchor,
            positive=positive,
            negative=negative,
        )

        # feat_anchor = F.normalize(_feat_anchor, dim=1)
        # feat_negative = F.normalize(_feat_negative, dim=1)
        # _feat_positive = F.normalize(_feat_positive, dim=1)

        # reshaping
        assert feat_anchor.shape[0] % 2 == 0

        # test1
        # feat_anchor = feat_anchor.view(feat_anchor.shape[0] // 2, 2, -1)
        # label_anchor = torch.ones(feat_anchor.shape[0], dtype=torch.long).to(
        #     feat_anchor.device
        # )
        # feat_negative = feat_negative.view(feat_negative.shape[0] // 2, 2, -1)
        # label_negative = torch.zeros(feat_negative.shape[0], dtype=torch.long).to(
        #     feat_negative.device
        # )
        # feats = torch.cat([feat_anchor, feat_negative], dim=0)
        # labels = torch.cat([label_anchor, label_negative], dim=0)

        # feat_zero = torch.zeros_like(feat_anchor).to(feat_anchor.device)
        # feat_one = torch.ones_like(feat_negative).to(feat_negative.device)
        # feats = torch.cat([feat_one, feat_zero], dim=0)

        # loss_contras = SupConLoss(temperature=0.2)(feats, labels)

        ## Test2
        # feats = torch.cat(
        #     [feat_anchor.unsqueeze_(1), feat_negative.unsqueeze_(1)], dim=1
        # )

        # loss_contras = SupConLoss(temperature=0.5)(feats)

        ## Test3
        # _feat_negative = feat_negative.unsqueeze(1)
        # loss_contras = self.criterion_infoNCE(
        #     feat_anchor, feat_positive, _feat_negative
        # )
        ## Test4
        _feat_negative = feat_negative.unsqueeze(1)
        loss_contras = self.criterion_infoNCE(feat_anchor, feat_anchor, _feat_negative)
        _feat_anchor = feat_anchor.unsqueeze(1)
        loss_contras += self.criterion_infoNCE(
            feat_negative, feat_negative, _feat_anchor
        )

        batch_size = anchor_output.size(0)

        # Generate random y values for each element in the batch
        y = torch.randint(0, 2, (batch_size,))

        # Initialize tensors to store the selected outputs
        logits = torch.empty_like(anchor_output).view(batch_size, -1)
        preds = torch.empty_like(anchor_output).view(batch_size, -1)

        # Iterate over the batch and select the corresponding output
        # 0 real 1 fake
        for i in range(batch_size):
            if y[i] == 0:
                logits[i] = anchor_output[i].flatten()
                preds[i] = anchor_output[i].sigmoid().flatten()
            else:
                logits[i] = negative_output[i].flatten()
                preds[i] = negative_output[i].sigmoid().flatten()

        # Ensure y is a float tensor and has the same shape as logits
        y = y.float().view(-1, 1).repeat(1, logits.shape[1]).to(logits.device)
        loss_cls = self.criterion_cls(logits, y)
        return loss_contras, loss_cls, logits, preds, y, feat_anchor, feat_negative

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        (
            loss_contras,
            loss_cls,
            logits,
            preds,
            y,
            _feat_anchor,
            _feat_negative,
        ) = self.model_step(batch)
        loss = (
            loss_contras * self.hparams.triplet_weight
            + loss_cls * self.hparams.cls_weight
        )

        # update and log metrics
        self.train_loss(loss)
        self.train_acc(logits, y)
        # self.train_ap(preds, targets)

        self.log(
            "train/contras_loss",
            loss_contras,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        self.log(
            "train/cls_loss", loss_cls, on_step=False, on_epoch=True, prog_bar=False
        )
        self.log(
            "train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log(
            "train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True
        )

        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        (
            loss_contras,
            loss_cls,
            logits,
            preds,
            y,
            _feat_anchor,
            _feat_negative,
        ) = self.model_step(batch)
        loss = (
            loss_contras * self.hparams.triplet_weight
            + loss_cls * self.hparams.cls_weight
        )

        # update and log metrics
        self.val_loss(loss)
        self.val_acc(logits, y)

        self.log(
            "val/triplet_loss",
            loss_contras,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        self.log("val/cls_loss", loss_cls, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

        # # viz features:
        pca_path, tsne_path = visualize_features(_feat_anchor, _feat_negative)
        # Ugly fix in pl for logging
        self.logger.experiment.log({"pca_res_pos": [wandb.Image(pca_path)]})
        self.logger.experiment.log({"tsne_res_pos": [wandb.Image(tsne_path)]})

        mean_inter, mean_intra_pos, mean_intra_neg = stat_feature(
            _feat_anchor, _feat_negative, norm_mode="cosine"
        )
        self.log(
            "stat/feat_dist_mean",
            mean_inter,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "stat/mean_intra_pos",
            mean_intra_pos,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "stat/mean_intra_neg",
            mean_intra_neg,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        # # viz features:
        # pca_pos_table, pca_neg_table = visualize_features(_feat_anchor, _feat_negative)
        # # Ugly fix in pl for logging
        # self.logger.experiment.log(
        #     {
        #         "pca_res_pos": [
        #             wandb.plot.scatter(
        #                 pca_pos_table, x="pca_x", y="pca_y", title="pca_pos"
        #             )
        #         ]
        #     }
        # )
        # self.logger.experiment.log(
        #     {"pca_res_neg": [wandb.plot.scatter(pca_neg_table, x="pca_x", y="pca_y")]}
        # )

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        acc = self.val_acc.compute()  # get current val acc
        self.val_acc_best(acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log(
            "val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True
        )

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss_triplet, loss_cls, preds, targets = self.model_step(batch)
        loss = loss_triplet + loss_cls

        # update and log metrics
        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.log(
            "test/triplet_loss",
            loss_triplet,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        self.log(
            "test/cls_loss", loss_cls, on_step=False, on_epoch=True, prog_bar=False
        )
        self.log(
            "test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = CoLaDDeModule(None, None, None, None)
