

<div align="center">

# tum-dlr-automl-for-eo
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
</div>

# Description
 Towards a NAS Benchmark for Classification in Earth Observation

# Quickstart

## Installation
- Create the pipeline environment and install the tum_dlr_automl_for_eo package
- Before using the template, one needs to install the project as a package.

* First, create a virtual environment.
> You can either do it with conda (preferred) or venv.
* Then, activate the environment
* Install the Naslib with the command below:
```
pip install -e git+https://github.com/emreds/NASLib.git#egg=naslib
```
* Then cd into the project's folder: 
```
cd tum-dlr-automl-for-eo
```
* Finally, install the rest of the dependencies Run:
```
pip install -e .
```

# How to Use? 
- Main functions to trigger are under the `./scripts` folder. 
- There are many scripts, including helper functions like `cluster_archs.py` which is not necessary for the main functionality. 
- `nb101_dict_creator.py` reads the pickle containing NB101 architectures and converts them into json dict format.
- `path_sampler.py` reads the NB101 dict and also the list of previously trained architectures from NB101(if any) and samples the new architures using random walk sampling.
- `bash_slurm` folder contains the bash scripts to submit training jobs to slurm using bash script. Every training job is submitted separately the have a certain level of fault tolerancy during the training. 
- `batch_train_submit.py` submits the training jobs using bash scripts in batch.
