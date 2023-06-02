import argparse
import os
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import torchvision.transforms as transforms
from torchmetrics.classification import MulticlassAccuracy
from training_torchlightning import LightningNetwork
from tum_dlr_automl_for_eo.datamodules.EODataLoader import EODataModule
from tum_dlr_automl_for_eo.utils import file

ARCHITECTURE_DIR = "/p/project/hai_nasb_eo/training/sampled_archs"
LOG_DIR = "/p/project/hai_nasb_eo/training/logs"

def get_params(args):
    return {
        "arch_path": args.arch,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay
    }
    
def print_model_params(model, limit):
    """
    Testing function to see model params.

    Args:
        model (_type_): _description_
        limit (_type_): _description_
    """
    i = 0
    for name, param in model.named_parameters():
        
        if i == limit: 
            break
        print(f"{name}: {param}")
        i += 1
    
def get_args():
    parser = argparse.ArgumentParser(
                    prog = "TUM-DLR-EO training script.",
                    description = "Trains the given model architecture.")
    parser.add_argument("--arch", required=False, help="Path of the architecture file.", default="/p/project/hai_nasb_eo/training/sampled_archs/arch_269")
    # just for once I will download the dataset into the permanent storage.
    parser.add_argument("--data", default="/p/project/hai_nasb_eo/data", help="Path of the training data.")
    parser.add_argument("--result", default="/p/project/hai_nasb_eo/training/logs", help="Path to save training results.")
    parser.add_argument("--batch_size", default=512, type=int, help= "Batch size should be divided by the number of gpus if ddp is enabled")
    parser.add_argument("--epoch", default=1, type=int)
    parser.add_argument("--lr", default=10e-5, type=float, help="learning rate should be scaled with the batch size \
        so that the sample variance of the gradients are approximately constant. \
        For DDP, it is scaled proportionally to the effective batch size, i.e. batch_size * num_gpus * num_nodes \
        For example, batch_size = 512, gpus=2, then lr = lr * sqrt(2) \
        Another suggestion is just use linear scaling from one of the most cited paper for DDP training: https://arxiv.org/abs/1706.02677")
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--num_workers", default=96, type=int)
    parser.add_argument("--weight_decay", default=5e-4, type=float)
    
    # training with GPU settings, including DDP
    parser.add_argument("--ddp", default=True, type=bool, help="Enable 1 node - multiple GPUs training")
    parser.add_argument("--gpus", default=4, type=int, help="Specify number of gpus used for training, given accelerator is gpu, can be >1 if ddp flag is enabled")
    parser.add_argument("--accelerator", default='gpu', type=str, help="Use different devices for training, e.g. gpu")
    parser.add_argument("--fast_dev_run", default=0, type=int, help="Test train/val/test pipeline by running a specific number of batches")
    
    args = parser.parse_args()
    
    return args

def prepare_test_data():
    """
    Prepares the testing data for the inference.

    Returns:
        test_data
    """
    
    test_data_mean = [
        1.278449594974517822e-01,
        1.149842068552970886e-01,
        1.111395284533500671e-01,
        1.232199594378471375e-01,
        1.645713448524475098e-01,
        1.862128973007202148e-01,
        1.792910993099212646e-01,
        2.002600133419036865e-01,
        1.727724820375442505e-01,
        1.278162151575088501e-01
        ]
    
    test_data_std = [
        3.514893725514411926e-02,
        4.023178666830062866e-02,
        5.523603409528732300e-02,
        5.091508477926254272e-02,
        6.154564023017883301e-02,
        7.297030836343765259e-02,
        7.590688019990921021e-02,
        8.254054188728332520e-02,
        8.815932273864746094e-02,
        8.100783824920654297e-02,
    ]
    
    data_module = EODataModule(args.data, "Sentinel-2")
    data_module.prepare_data()
    
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(test_data_mean, test_data_std)
        ]
    )
    data_module.setup_testing_data(test_transform)
    test_data = data_module.testing_dataLoader(
                batch_size=args.batch_size, num_workers=0
            )
    
    return test_data

def calculate_accuracy(model, test_data):
        
        get_macro_acc = MulticlassAccuracy(
                num_classes=17, average="macro"
            )
        get_micro_acc = MulticlassAccuracy(
                num_classes=17, average="micro"
            )
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for images, targets in test_data: 
                outputs = model(images.float())
                values, preds = outputs.max(1)
                
                all_preds.append(preds)
                all_targets.append(targets)

            all_preds = torch.cat(all_preds)
            all_targets = torch.cat(all_targets)
            
            micro_acc = get_micro_acc(all_preds, all_targets)
            macro_acc = get_macro_acc(all_preds, all_targets)
            
            print(f"Micro Accuracy: {micro_acc}")
            print(f"Macro Accuracy: {macro_acc}")
            
            print("it is here")


def calculate_normal(dataloader):
        # Calculate the mean and standard deviation
        mean = 0.0
        std = 0.0
        total_samples = 0

        for data, _ in dataloader:
            batch_samples = data.size(0)
            data = data.view(batch_samples, data.size(1), -1)
            mean += data.mean(2).sum(0)
            total_samples += batch_samples

        mean /= total_samples
        
        pixel_cnt = 0
        var = 0
        for data, _ in dataloader:
            batch_samples = data.size(0)
            data = data.view(batch_samples, data.size(1), -1)
            var += torch.square(data - mean.unsqueeze(1)).sum([0,2])
            pixel_cnt += data.nelement()
        
        #std /= total_samples
        std = torch.sqrt(var / pixel_cnt)
        print("Mean:", mean)
        print("Std:", std)
        
        return mean, std
    
def get_checkpoints(log_dir: Path, exp_number:str = "0_0", epoch:str ="107"):
    
    train_logs = os.listdir(log_dir)
    epoch_prefix = "epoch=" + epoch
    
    arch_checkpoint = {}
    for dir_name in train_logs:
        word_list = dir_name.split("_")
        # Network checkpoints are saved in format `arch_x_arch_x`.
        if len(word_list) > 2:
            arch_code = word_list[0] + '_' + word_list[1]
            checkpoint_dir = log_dir / dir_name / "0_0" / "checkpoints"
            epoch = [epoch_path for epoch_path in os.listdir(checkpoint_dir) if epoch_path.split("-")[0] == epoch_prefix]
            if epoch:
                arch_checkpoint[arch_code] = checkpoint_dir / epoch[0]
            else:
                continue
            
    return arch_checkpoint



def load_architecture():
    args = get_args()
    params = get_params(args)
    arch_dir = Path(ARCHITECTURE_DIR)
    log_dir = Path(LOG_DIR)
    archs = file.get_arch_paths(arch_dir)
    arch_checkpoint = get_checkpoints(log_dir)
    arch_paths = sorted([str(path) for path in archs])[:-1] # we exclude arch_specs.json
    arch_param = {}
    for path in arch_paths:
        arch_code = path.split('/')[-1]
        if arch_code in arch_checkpoint:
            checkpoint = arch_checkpoint[arch_code]
            params["arch_path"] = path
            arch_param[checkpoint] = {"params": params}
        
    return arch_param
        
    """
    checkpoint = torch.load("/p/project/hai_nasb_eo/training/logs/arch_269_arch_269/0_0/checkpoints/epoch=107-step=148715.ckpt", map_location=torch.device('cpu'))
    model = LightningNetwork(params)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    """
    
    pass
    

if __name__ == "__main__": 
    #cd torch.device('cpu')
    
    model = load_architecture()
    
    #print(checkpoint.keys())
    
    #test_data = prepare_test_data()
    #calculate_accuracy(model, test_data)
    