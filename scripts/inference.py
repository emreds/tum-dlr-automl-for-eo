import json
import os
import pathlib
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
import torchvision.transforms as transforms
import training_torchlightning
from thop import clever_format, profile
from torchmetrics.classification import MulticlassAccuracy
from tum_dlr_automl_for_eo.datamodules.EODataLoader import EODataModule
from tum_dlr_automl_for_eo.utils import file

unmatched_archs = []

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


def prepare_test_data(data_dir, batch_size, num_workers):
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
    
    data_module = EODataModule(data_dir, "Sentinel-2")
    data_module.prepare_data()
    
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(test_data_mean, test_data_std)
        ]
    )
    data_module.setup_testing_data(test_transform)
    test_data = data_module.testing_dataLoader(
                batch_size=batch_size, num_workers=num_workers
            )
    
    return test_data


    

class CheckpointParams: 
    def __init__(self, args, arch_dir, log_dir, batch_size, exp_number="0_0", epoch="107"):
        self.args = args
        self.batch_size = batch_size
        self.args.batch_size = self.batch_size
        self.arch_dir = pathlib.Path(arch_dir)
        self.log_dir = pathlib.Path(log_dir)
        
        self.exp_number = exp_number
        self.epoch = epoch
        self.base_archs = file.get_base_arch_paths(self.arch_dir)
        if not exp_number:
            self.checkpoints = file.get_checkpoint_paths_clean(self.log_dir, epoch)
        else:
            self.checkpoints = file.get_checkpoint_paths(self.log_dir, exp_number, epoch)    
    
    
    def load_params(self):
        #arch_paths = sorted([str(path) for path in self.base_archs])[:-1] # we exclude arch_specs.json
        arch_paths = sorted([str(path) for path in self.base_archs])
        check_param = {}
        for path in arch_paths:
            arch_code = path.split('/')[-1].split('.')[0]
            if arch_code in self.checkpoints:
                checkpoint = self.checkpoints[arch_code]
                check_param[checkpoint] = {"params": self.get_params(path), "arch_code": arch_code}
            else: 
                unmatched_archs.append(arch_code)
        return check_param

    def get_params(self, arch_path:str):
        return {
            "arch_path": arch_path,
            "batch_size": self.batch_size,
            "lr": self.args.lr,
            "momentum": self.args.momentum,
            "weight_decay": self.args.weight_decay
        }

class TestArch:
    
    def __init__(self, checkpoint_path:str, dataloder:torch.utils.data.DataLoader, params:dict) -> None:
        self.checkpoint_path = checkpoint_path
        self.params = params
        self.arch = None 
        self.test_data = dataloder
        self.accuracy = {"micro": 0.0, "macro": 0.0}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.get_macro_acc = MulticlassAccuracy(
                num_classes=17, average="macro"
            ).to(self.device)
        self.get_micro_acc = MulticlassAccuracy(
                num_classes=17, average="micro"
            ).to(self.device)
        
        self.avg_inference_time = 0.0
        self.std_inference_time = 0.0
        self.num_params = 0.0
        self.arch_size = 0.0
        self.allocated_memory = 0.0
        self.macs = 0.0
        self.results = {"accuracy": self.accuracy,
                        "avg_inference_time": self.avg_inference_time,
                        "num_params": self.num_params,
                        "arch_size": self.arch_size, 
                        "MACs": self.macs,
                        }
        
    def __call__(self):
        self.arch_size = self.file_size(self.checkpoint_path)
        self.arch = self.load_architecture(self.checkpoint_path, self.params)
        self.num_params = sum(p.numel() for p in self.arch.parameters())
        self.calculate_macs()
        self.calculate_accuracy()
        
        self.results = {"accuracy": self.accuracy,
                        "avg_inference_time": self.avg_inference_time,
                        "std_inference_time": self.std_inference_time,
                        "num_params": self.num_params,
                        "arch_size": self.arch_size, 
                        "MACs": self.macs
                        }
        
        return self.results

    
    def file_size(self, file_path, unit='kb'):
        """
        Calculates the file size of a given file.

        Args:
            file_path (_type_): _description_
            unit (str, optional): Choose one: ['bytes', 'kb', 'mb', 'gb']. Defaults to 'kb'.

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        file_size = os.path.getsize(file_path)
        exponents_map = {'bytes': 0, 'kb': 1, 'mb': 2, 'gb': 3}
        if unit not in exponents_map:
            raise ValueError("Must select from \
            ['bytes', 'kb', 'mb', 'gb']")
        else:
            size = file_size / 1024 ** exponents_map[unit]
            return round(size, 6)
        
        
    def calculate_accuracy(self):
        
        all_preds = []
        all_targets = []

        if self.device.type == "cuda":
            dummy_input = torch.randn(torch.Size([64,10,32,32]), dtype=torch.float).to(self.device)
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            repetitions = len(self.test_data)
            timings=np.zeros((repetitions,1))
            #GPU-WARM-UP
            for _ in range(10):
                _ = self.arch(dummy_input)
            
            with torch.no_grad():
                i = 0
                for images, targets in self.test_data: 
                    starter.record()
                    outputs = self.arch(images.float().to(self.device))
                    ender.record()
                    
                    # WAIT FOR GPU SYNC
                    torch.cuda.synchronize()
                    curr_time = starter.elapsed_time(ender)
                    timings[i] = curr_time
                    i += 1
                    values, preds = outputs.max(1)
                    all_preds.append(preds)
                    all_targets.append(targets)

                all_preds = torch.cat(all_preds).to(self.device)
                all_targets = torch.cat(all_targets).to(self.device)
                
                self.accuracy["micro"] = self.get_micro_acc(all_preds, all_targets).cpu().tolist()
                self.accuracy["macro"] = self.get_macro_acc(all_preds, all_targets).cpu().tolist()
            
            self.avg_inference_time = np.sum(timings) / repetitions
            self.std_inference_time = np.std(timings)
            
        return self.accuracy
    
    def load_architecture(self, checkpoint_path:str, params: dict):
        """
        Loads the architecture from the given checkpoint path.
        Args:
            checkpoint_path: Path to the checkpoint.
            params: Parameters for the architecture.

        Returns:
            model: The architecture.
        """
        if self.device.type == "cuda":
            before_mem = torch.cuda.memory_allocated(self.device)
            
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model = training_torchlightning.LightningNetwork(params)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        model.to(self.device)
        
        if self.device.type == "cuda":
            after_mem = torch.cuda.memory_allocated(self.device)
            self.allocated_memory = after_mem - before_mem
        
        return model
    
    def calculate_macs(self):
        input = torch.randn([64,10,32,32]).to(self.device)          
        macs, params = profile(self.arch, inputs=(input, ), verbose=False)
        macs, params = clever_format([macs, params], "%.3f")
        
        self.macs = params
        
        return self.macs

if __name__ == "__main__": 
    ARCH_DIR = "/p/project/hai_nasb_eo/sampled_paths/untrained_archs"
    LOG_DIR = "/p/project/hai_nasb_eo/sampled_paths/all_trained_archs"

    args = training_torchlightning.get_args(require_arch=False)
    test_dataloader = prepare_test_data(args.data, args.batch_size, num_workers=0)
    check_param = CheckpointParams(args=args, arch_dir=ARCH_DIR, log_dir=LOG_DIR, batch_size=args.batch_size, exp_number='', epoch="107")
    arch_params = check_param.load_params()
    #print(unmatched_archs)
    #print(arch_params)

    results = {}
    
    for arch_path, vals in arch_params.items():
        tests = TestArch(arch_path, test_dataloader, vals["params"])()
        tests["checkpoint_path"] = arch_path
        results[vals["arch_code"]] = tests
        
        
    print(results)

    class PosixPathEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, pathlib.PurePosixPath):
                return str(obj)
            return json.JSONEncoder.default(self, obj)
    json_str = json.dumps(results, cls=PosixPathEncoder)

    with open("./test_results_sampled_all_paths", "w") as f:
        f.write(json_str)
    