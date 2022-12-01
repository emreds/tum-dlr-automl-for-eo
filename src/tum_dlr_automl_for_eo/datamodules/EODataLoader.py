from torch.utils.data import DataLoader
import logging as log
from pytorch_lightning import LightningDataModule
from So2SatDataSet import So2SatDataSet
from monai.apps import download_url
import os, zipfile
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

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

    def setup_training_data(self, transform=None) -> None:
        path_to_read_data = self.path_to_download_data + "/So2SatLCZ/data/m1483140"
        self.training_data = So2SatDataSet(path_to_read_data + '/training.h5', self.sentinel,
                                           transform=transform)
        log.info(f"Number of training data: {len(self.training_data)}")
        log.info(f"Number of classes: {self.num_classes}")

    def setup_validation_data(self, transform=None) -> None:
        path_to_read_data = self.path_to_download_data + "/So2SatLCZ/data/m1483140"
        self.validation_data = So2SatDataSet(path_to_read_data + '/validation.h5', self.sentinel,
                                             transform=transform)
        log.info(f"Number of validation data: {len(self.validation_data)}")
        log.info(f"Number of classes: {self.num_classes}")

    def setup_testing_data(self, transform=None) -> None:
        path_to_read_data = self.path_to_download_data + "/So2SatLCZ/data/m1483140"
        self.testing_data = So2SatDataSet(path_to_read_data + '/testing.h5', self.sentinel,
                                          transform=transform)

        log.info(f"Number of testing data: {len(self.testing_data)}")
        log.info(f"Number of classes: {self.num_classes}")

    def training_dataLoader(self,
                            batch_size: int = 64,
                            pin_memory: bool = True,
                            # num_workers: int = 0,
                            ):
        if self.training_data is not None:
            return DataLoader(self.training_data, batch_size, pin_memory)
        else:
            raise Warning("training data is None, did you setup it ?")

    def validation_dataLoader(self,
                              batch_size: int = 64,
                              pin_memory: bool = True,
                              # num_workers: int = 0,
                              ):
        if self.validation_data is not None:
            return DataLoader(self.validation_data, batch_size, pin_memory)
        else:
            raise Warning("validation data is None, did you set it up ?")

    def testing_dataLoader(self,
                           batch_size: int = 64,
                           pin_memory: bool = True,
                           # num_workers: int = 0,
                           ):
        if self.testing_data is not None:
            return DataLoader(self.testing_data, batch_size, pin_memory)
        else:
            raise Warning("testing data is None, did you set it up ?")
