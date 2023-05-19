import argparse

import pytorch_lightning as pl
import torch
import torchvision.transforms as transforms
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
    
def get_args():
    parser = argparse.ArgumentParser(
                    prog = "TUM-DLR-EO training script.",
                    description = "Trains the given model architecture.")
    parser.add_argument("--arch", required=False, help="Path of the architecture file.", default="")
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

args = get_args()
params = get_params(args)
# Instantiate the dummy LightningModule
checkpoint = torch.load(args.arch)
model = LightningNetwork(params)

model.eval()

data_module = EODataModule(args.data, "Sentinel-2")
data_module.prepare_data()
test_transform = transforms.Compose(
    [
        transforms.ToTensor()
    ]
)
data_module.setup_testing_data(test_transform)
testing_data = data_module.testing_dataLoader(
            batch_size=args.batch_size, num_workers=80
        )
predictions = []

for name, param in model.named_parameters(): 
    print(f"{name}: {param}")

'''
with torch.no_grad(): 
    for images, _ in testing_data: 
        outputs = model(images)
        print(outputs)
'''