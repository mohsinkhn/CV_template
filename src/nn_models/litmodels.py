from typing import Any, Dict
import lightning as L
import torch
from torch import nn
import torch.nn.functional as F


class BaseLitModel(L.LightningModule):
    """PL Model"""

    def __init__(
        self,
        model: nn.Module,
        loss_fn: Any,
        optimizer: Any,
        scheduler: torch.optim.lr_scheduler,
        scheduler_interval: str,
        compile: bool,
        label_field: str = "label",
        data_field: str = "image_data",
        val_output_dir: str = "./data",
        test_output_dir: str = "./data",
    ):
        super().__init__()
        self.save_hyperparameters(
            logger=False, ignore=["model", "loss_fn", "optimizer", "scheduler"]
        )
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, x):
        return self.model(x[self.hparams.data_field])

    def step(self, batch):
        y = batch[self.hparams.label_field]
        logits = self.forward(batch)
        loss = self.criterion(logits, y)
        return loss, logits, y

    def training_step(self, batch, batch_idx):
        loss, _, _ = self.step(batch)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, y = self.step(batch)
        self.post_process_validation_step(loss, preds, y, batch, batch_idx)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        logits = self.forward(batch)
        self.post_process_test_step(logits, batch, batch_idx)

    def on_validation_epoch_end(self) -> None:
        data = {}
        for batch_data in self.validation_step_outputs:
            for k, v in batch_data.items():
                if k not in data:
                    data[k] = []
                data[k].append(v.cpu())
        for k, v in data.items():
            data[k] = torch.cat(v, dim=0).numpy()
        self.post_process_validation_epoch_end(data)
        self.validation_step_outputs = []

    def on_test_epoch_end(self) -> None:
        data = {}
        for batch_data in self.test_step_outputs:
            for k, v in batch_data.items():
                if k not in data:
                    data[k] = []
                data[k].append(v.cpu())
        for k, v in data.items():
            data[k] = torch.cat(v, dim=0).numpy()
        self.post_process_test_epoch_end(data)
        self.test_step_outputs = []

    def predict_step(self, batch, batch_idx):
        _, preds, _ = self.step(batch)
        return preds

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.model = torch.compile(self.model)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.optimizer(params=self.trainer.model.parameters())
        if self.scheduler is not None:
            scheduler = self.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": self.hparams.scheduler_interval,
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

    def post_process_validation_step(
        self, loss, preds, y, batch: Dict[str, Any], batch_idx: int
    ) -> None:
        pass

    def post_process_test_step(
        self, preds, batch: Dict[str, Any], batch_idx: int
    ) -> None:
        pass

    def post_process_validation_epoch_end(self, data) -> None:
        pass
