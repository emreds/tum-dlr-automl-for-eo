

<div align="center">

# tum-dlr-automl-for-eo
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/HelmholtzAI-Consultants-Munich/ML-Pipeline-Template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a>
<a href="https://github.com/pyscaffold/pyscaffoldext-dsproject"><img alt="Template" src="https://img.shields.io/badge/-Pyscaffold--Datascience-017F2F?style=flat&logo=github&labelColor=gray"></a>

</div>

# Description
 Towards a NAS Benchmark for Classification in Earth Observation

# Quickstart

## Create the pipeline environment and install the tum_dlr_automl_for_eo package
Before using the template, one needs to install the project as a package.
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
