import argparse
import logging
import torch

from tum_dlr_automl_for_eo.datamodules.EODataLoader import EODataModule

import torchvision.transforms as transforms
import torchvision.datasets as dset
from pathlib import Path
import torch.nn.functional as F
from tqdm import tqdm
from time import sleep

import os

def get_args():
    parser = argparse.ArgumentParser(
                    prog = "TUM-DLR-EO training script.",
                    description = "Trains the given model architecture.")
    parser.add_argument("--arch", required=True, help="Path of the architecture file.")
    # just for once I will download the dataset into the permanent storage.
    parser.add_argument("--data", default="/p/project/hai_nasb_eo/data", help="Path of the training data.")
    parser.add_argument("--result", default="/p/project/hai_nasb_eo/training/logs", help="Path to save training results.")
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--epoch", default=1, type=int)
    parser.add_argument("--lr", default=0.4, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--weight_decay", default=5e-4, type=float)

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

def train_loop(network, training_data, validation_data, args):
    epoch = args.epoch

    
    optimizer = torch.optim.SGD(network.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    # Not sure if we are going to use the scheduler.
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    best_val_acc = 0.0
    network.train()
    for epoch in range(1, epoch+1):
        with tqdm(training_data, unit="batch") as tepoch:
            for data, target in tepoch:
                tepoch.set_description(f"Epoch {epoch}")
                
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = network(data.float())
                predictions = output.argmax(dim=1, keepdim=True).squeeze()
                loss = F.nll_loss(output, target)
                correct = (predictions == target).sum().item()
                accuracy = correct / batch_size
                
                loss.backward()
                optimizer.step()

                tepoch.set_postfix(loss=loss.item(), accuracy=100. * accuracy)
                sleep(0.1)
        
                logging.info(f"Epoch-{epoch}, Training mean loss: {loss}, Mean accuracy: {accuracy}")
                
        #scheduler.step()
        network.eval()
        val_loss = 0.0
        val_acc = 0.0   
        for data, target in validation_data:
            output = network(data.float)
            predictions = output.argmax(dim=1, keepdim=True).squeeze()
            loss = F.nll_loss(output, target)
            val_loss += loss
            correct = (predictions == target).sum().item()
            accuracy = correct / batch_size
            val_acc += accuracy
        
        val_loss /= len(validation_data)
        val_acc /= len(validation_data)
        if val_acc > best_val_acc: 
            best_val_acc = val_acc
        logging.info(f"Epoch-{epoch}, Validation mean loss: {val_loss}, Validation mean accuracy: {val_acc}")
    
    logging.info(f"Training completed successfully for {epoch} epochs.")
    logging.info(f"Best validation accuracy: {best_val_acc}")
    
    
if __name__ == "__main__":
    args = get_args()
    data_path = args.data
    arch_path = args.arch
    result_path = args.result
    
    print('IT was able to get the args.')
    try: 
        network = torch.load(arch_path)
        print('after torch load')
        batch_size = args.batch_size
        print('after batch size')
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        train_transform, valid_transform = get_data_transforms()
        
        data_module = EODataModule(data_path, 'Sentinel-2')
        data_module.prepare_data()

        data_module.setup_training_data(train_transform)
        training_data = data_module.training_dataLoader(batch_size = batch_size)

        data_module.setup_validation_data(valid_transform)
        validation_data = data_module.validation_dataLoader(batch_size = batch_size)
        
        #data_module.setup_testing_data()
        
        train_loop(network, training_data, validation_data, args)
    except Exception as e:
        logging.error(f"During training some error occured, error: {e}")
