from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from methazAI.datasets.amazon_utils import ensemble, clean_data
import gzip, pickle
from hydra.utils import get_original_cwd
from typing import Optional
from omegaconf import DictConfig
from datasets import load_dataset

# This is hackidy-hacky:
import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"
# Hackidy hack over ¯\_(ツ)_/¯


class AmazonDataModule(LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.config = cfg.datamodule
        self.wd = get_original_cwd()
        self.prepare_data()

    def prepare_data(self):
        ensemble(self.config, self.wd)
        clean_data(self.config, self.wd)

    def setup(self, stage: Optional[str] = None):
        # called on every GPU
        datasets = self.load_datasets(
            f'{self.wd}/data/SA_amazon_data/processed/{self.config["name"]}/'
        ).with_format("torch")
        self.train = datasets["train"]
        self.val = datasets["val"]
        self.test = datasets["test"]
        self.t_total = len(self.train) // self.config["batch_size"]

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.config["batch_size"],
            num_workers=self.config["num_workers"],
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.config["batch_size"],
            num_workers=self.config["num_workers"],
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.config["batch_size"],
            num_workers=self.config["num_workers"],
        )

    def load_datasets(self, folder_path):
        try:
            datasets = load_dataset(
                "parquet",
                data_files={
                    "train": f"{folder_path}train.parquet",
                    "val": f"{folder_path}val.parquet",
                    "test": f"{folder_path}test.parquet",
                },
            )
            return datasets
        except Exception as ex:
            if type(ex) == FileNotFoundError:
                raise FileNotFoundError(
                    f"The datasets could not be found in {folder_path}"
                )
