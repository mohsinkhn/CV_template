from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from lightning import LightningDataModule
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset

from src.nn_datasets import components


class BaseDataModule(LightningDataModule):
    def __init__(
        self,
        data_csv: Path,
        train_dataset: Dataset,
        val_dataset: Dataset,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        test_batch_size_multiplier: int = 1,
        test_csv: Path = None,
        test_dataset: Dataset = None,
    ):
        super().__init__()
        self.save_hyperparameters(
            logger=False, ignore=["train_dataset", "val_dataset", "test_dataset"]
        )

        self.data_train = train_dataset
        self.data_val = val_dataset
        self.data_test = test_dataset

        self.batch_size_per_device = batch_size

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        self.df = pd.read_csv(self.hparams.data_csv)
        self.df = self._add_fold_ids(self.df)

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

        if stage == "fit" or stage is None:
            self.data_train, self.data_val = self._prepare_data()

        if stage == "test" or stage is None:
            self.data_test = self._prepare_test_data()

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device
            * self.hparams.test_batch_size_multiplier,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            drop_last=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device
            * self.hparams.test_batch_size_multiplier,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            drop_last=False,
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
        pass

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass

    def _prepare_data(self) -> Tuple[Dataset, Dataset]:
        """Prepare the training and validation data.

        :return: A tuple with the training and validation datasets.
        """
        train_df = self.df[self.df["fold"] != self.hparams.fold]
        val_df = self.df[self.df["fold"] == self.hparams.fold]

        train_dataset = self.data_train(df=train_df)
        val_dataset = self.data_val(df=val_df)
        return train_dataset, val_dataset

    def _prepare_test_data(self) -> Dataset:
        """Prepare the test data.

        :return: The test dataset.
        """
        test_df = pd.read_csv(self.hparams.test_csv)
        return self.data_test(df=test_df)

    def _add_fold_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add fold ids to the dataframe.

        :param df: The dataframe to which the fold ids will be added.
        :return: The dataframe with the fold ids.
        """
        cvlist = KFold(
            n_splits=self.hparams.num_folds, shuffle=True, random_state=42
        ).split(df)
        df["fold"] = -1
        for fold, (train_idx, val_idx) in enumerate(cvlist):
            df.loc[val_idx, "fold"] = fold
        return df
