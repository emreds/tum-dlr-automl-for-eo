import argparse

import numpy as np
import pytorch_lightning as pl
import torch
import torchvision.transforms as transforms
from torchmetrics.classification import MulticlassAccuracy
from training_torchlightning import LightningNetwork
from tum_dlr_automl_for_eo.datamodules.EODataLoader import EODataModule


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
    test_data_mean = [0.1278, 0.1149, 0.1110, 0.1230, 0.1643, 0.1859, 0.1790, 0.1999, 0.1724,
        0.1275
        
    ]
    
    test_data_std = [0.0151, 0.0180, 0.0251, 0.0209, 0.0265, 0.0313, 0.0366, 0.0348, 0.0329,
        0.0313]

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


def calculate_normal(dataloader):
        # Calculate the mean and standard deviation
        mean = 0.0
        std = 0.0
        total_samples = 0

        for data, _ in dataloader:
            batch_samples = data.size(0)
            data = data.view(batch_samples, data.size(1), -1)
            mean += data.mean(2).sum(0)
            std += data.std(2).sum(0)
            total_samples += batch_samples

        mean /= total_samples
        std /= total_samples

        print("Mean:", mean)
        print("Std:", std)
        
        return mean, std
    

if __name__ == "__main__": 
    torch.device('cpu')
    
    args = get_args()
    params = get_params(args)
    checkpoint = torch.load("/p/project/hai_nasb_eo/training/logs/arch_269_arch_269/0_0/checkpoints/epoch=107-step=148715.ckpt", map_location=torch.device('cpu'))
    model = LightningNetwork(params)
    #print(checkpoint.keys())
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    #predictions = torch.tensor([])
    test_data = prepare_test_data()
    calculate_accuracy(model, test_data)
    