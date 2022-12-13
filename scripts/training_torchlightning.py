import argparse
import logging
import torch

import torchvision.transforms as transforms
import torch.nn.functional as F
import torch
import pytorch_lightning as pl

from tum_dlr_automl_for_eo.datamodules.EODataLoader import EODataModule
import os
import time 

import logging

class CustomCallback(pl.Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        print("\nTraining epoch starts")
        self.time = time.time()
    
    def on_train_epoch_end(self, trainer, pl_module): 
        logging.info(f"\nTraining epoch ends, training time: {self.time - time.time()}")
        self.time = 0

class LightningNetwork(pl.LightningModule):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.network = torch.load(params["arch_path"])
    
    def forward(self, x):
        return self.network.forward(x)
    
    def cross_entropy_loss(self, logits, labels):
        return F.nll_loss(logits, labels)

    def training_step(self, train_batch, batch_idx):
        data, targets = train_batch 
        logits = self.forward(data.float())     

        # predictions 
        predictions = logits.argmax(dim=1, keepdim=True).squeeze()
        correct = (predictions == targets).sum().item()
        accuracy = correct / self.params["batch_size"]

        # loss
        loss = self.cross_entropy_loss(logits, targets)

        logging.info(f"train_accuracy: {accuracy}, train_loss: {loss}")

        # test lightning loss
        self.log("train accuracy: ", accuracy)

        return loss
    
    def validation_step(self, val_batch, batch_idx):
        data, targets = val_batch 
        logits = self.forward(data.float())

        # same logic
        predictions = logits.argmax(dim=1, keepdim=True).squeeze()
        correct = (predictions == targets).sum().item()
        accuracy = correct / self.params["batch_size"]
        loss = self.cross_entropy_loss(logits, targets)

        logging.info(f"val_accuracy: {accuracy}, val_loss: {loss}")
        
        # test lightning log
        self.log("test accuracy: ", accuracy)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.network.parameters(), 
                                    lr=self.params["lr"], 
                                    momentum=self.params["momentum"], 
                                    weight_decay=self.params["weight_decay"])

        # optimizer = torch.optim.Adam(
        #                 self.network.parameters(),
        #                 lr = self.params["lr"],
        #                 weight_decay = self.params["weight_decay"]
        #             )

        return optimizer

def get_args():
    parser = argparse.ArgumentParser(
                    prog = "TUM-DLR-EO training script.",
                    description = "Trains the given model architecture.")
    parser.add_argument("--arch", required=True, help="Path of the architecture file.")
    # just for once I will download the dataset into the permanent storage.
    parser.add_argument("--data", default="/p/project/hai_nasb_eo/data", help="Path of the training data.")
    parser.add_argument("--result", default="/p/project/hai_nasb_eo/training/logs", help="Path to save training results.")
    parser.add_argument("--batch_size", default=512, type=int)
    parser.add_argument("--epoch", default=1, type=int)
    parser.add_argument("--lr", default=0.1, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--num_workers", default=96, type=int)
    parser.add_argument("--weight_decay", default=0, type=float)
    
    # ddp settings
    parser.add_argument("--gpus", default=1, type=int)
    
    args = parser.parse_args()
    
    return args

def set_logger(arch_path, result_path):
    '''
        Defined the log path.
    '''
    
    log_file = arch_path.split('/')[-1] + ".log"
    log_path = os.path.join(result_path, log_file)
    logging.basicConfig(filename=log_path, encoding='utf-8', level=logging.DEBUG)
    
    pass

def get_data_transforms():
    training_data_mean = [
    1.237384229898452759e-01,1.092514395713806152e-01,1.010472476482391357e-01,
    1.141963005065917969e-01,1.592123955488204956e-01,1.814149618148803711e-01,
    1.745131611824035645e-01,1.949533522129058838e-01,1.542119830846786499e-01,
    1.089953780174255371e-01]
    training_data_std = [
    3.955886885523796082e-02,4.774657264351844788e-02,6.632962822914123535e-02,
    6.356767565011978149e-02,7.745764404535293579e-02,9.104172885417938232e-02,
    9.217569977045059204e-02,1.016793847084045410e-01,9.986902773380279541e-02,
    8.776713907718658447e-02
    ]
    validation_data_mean = [
    1.289583146572113037e-01,1.164120137691497803e-01,1.122049614787101746e-01,
    1.240097358822822571e-01,1.646174490451812744e-01,1.862829625606536865e-01,
    1.791501641273498535e-01,2.004020363092422485e-01,1.738285273313522339e-01,
    1.277212053537368774e-01]
    validation_data_std = [
    3.845561295747756958e-02,4.351711273193359375e-02,5.868862569332122803e-02,
    5.489562824368476868e-02,6.446107476949691772e-02,7.579671591520309448e-02,
    7.847409695386886597e-02,8.553111553192138672e-02,9.418702870607376099e-02,
    8.428058028221130371e-02
    ]
    
    train_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(training_data_mean, training_data_std),
    ]
    )

    valid_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(validation_data_mean, validation_data_std),
        ]
    )
    
    return train_transform, valid_transform


def get_params(args):
    return {
        "arch_path": args.arch,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
    }

if __name__ == "__main__":
    args = get_args()
    data_path = args.data
    arch_path = args.arch
    result_path = args.result
    
    try: 
        params = get_params(args)
        network = LightningNetwork(params)

        batch_size = args.batch_size
        num_workers = args.num_workers
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print('currently using {} for training'.format(device))
        train_transform, valid_transform = get_data_transforms()
        
        data_module = EODataModule(data_path, 'Sentinel-2')
        data_module.prepare_data()

        data_module.setup_training_data(train_transform)
        training_data = data_module.training_dataLoader(batch_size = batch_size, num_workers=num_workers)

        data_module.setup_validation_data(valid_transform)
        validation_data = data_module.validation_dataLoader(batch_size = batch_size, num_workers=num_workers)
        
        #data_module.setup_testing_data()
        
        # lightning train
        trainer = pl.Trainer(
            devices = args.gpus,
            accelerator = "gpu",
            max_epochs = args.epoch,
            callbacks =[CustomCallback()]
        )
        trainer.fit(network, 
            train_dataloaders=training_data, 
            val_dataloaders=validation_data)
    except Exception as e:
        logging.error(f"During training some error occured, error: {e}")
