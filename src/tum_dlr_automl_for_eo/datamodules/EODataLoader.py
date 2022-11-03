from torch.utils.data import DataLoader
import logging as log
from pytorch_lightning import LightningDataModule
from So2SatDataSet import So2SatDataSet
from monai.apps import download_url
import os, zipfile


class EODataModule(LightningDataModule):

    def __init__(self, file_path, sentinel):
        super().__init__()
        self.path_to_download_data = file_path  ##existing path or path to download data
        self.sentinel = sentinel
        self.training_data = None
        self.validation_data = None
        self.testing_data = None

    @property
    def num_classes(self) -> int:
        return 17

    def prepare_data(self) -> None:
        data_dir = os.path.join(self.path_to_download_data, "So2SatLCZ")
        data_dir_files = os.path.join(self.path_to_download_data, "So2SatLCZ/data")

        resource = "https://dataserv.ub.tum.de/s/m1483140/download"
        compressed_file = os.path.join(data_dir, "So2SatLCZ_version2.tar")
        if not os.path.exists(compressed_file):
            download_url(url=resource, filepath=compressed_file)

        if not os.path.exists(data_dir_files):
            with zipfile.ZipFile(compressed_file, 'r') as banking_zip:
                banking_zip.extractall(data_dir_files)

        log.info(f"Data preparation is done, next step is data set-up")

    def setup(self, transform=None) -> None:
        self.path_to_download_data += "/So2SatLCZ/data/m1483140"
        transform = transform

        self.training_data = So2SatDataSet(self.path_to_download_data + '/training.h5', self.sentinel,
                                           transform=transform)
        self.validation_data = So2SatDataSet(self.path_to_download_data + '/validation.h5', self.sentinel,
                                             transform=transform)
        self.testing_data = So2SatDataSet(self.path_to_download_data + '/testing.h5', self.sentinel,
                                          transform=transform)

        log.info(f"Number of training data: {len(self.training_data)}")
        log.info(f"Number of testing data: {len(self.testing_data)}")
        log.info(f"Number of validation data: {len(self.validation_data)}")
        log.info(f"Number of classes: {self.num_classes}")

    def training_dataLoader(self,
                            batch_size: int = 1000,
                            ):
        return DataLoader(self.training_data, batch_size)

    def validation_dataLoader(self,
                              batch_size: int = 1000,
                              ):
        return DataLoader(self.validation_data, batch_size)

    def testing_dataLoader(self,
                           batch_size: int = 1000,
                           ):
        return DataLoader(self.testing_data, batch_size)
