# Simulation of Fluid Flows based on Data-driven Evolution of Vortex Particles

## Files 

- [Weekly Work Schedule](https://docs.google.com/spreadsheets/d/171NmDXxsqH2n5pfHbGNPH_iO45OriolQQLX5hPMuMYU/edit?usp=sharing)
- [Report](https://www.overleaf.com/4688964232sncxgbqgphzp)
- [Important Observed Issues and choosen Solutions](https://www.overleaf.com/read/vcjgdvstccmw)

## Installation

This repository is built on top of the differentiable fluid simulation solver [PhiFlow](https://github.com/tum-pbs/PhiFlow), which being used as a submodule inside this repository.
- The following instructions would apply for a system with **CUDA** enabled GPU with a **CUDA** version of **10.0**.  
- First clone the contents of this repository including [PhiFlow](https://github.com/tum-pbs/PhiFlow'). 
```
git clone --recurse-submodules https://github.com/VemburajYadav/DeepLearningLagrangainVortexDynamics.git
```
- The detailed instructions regarding the installation of **PhiFlow** could be found [here](https://github.com/tum-pbs/PhiFlow/blob/master/documentation/Installation_Instructions.md). Following steps could be directly used to install [PhiFlow](https://github.com/tum-pbs/PhiFlow') once the repository is cloned when setting up with **anaconda**.
```
$ conda create -n tf python=3.6
$ conda activate tf
$ pip install tensorflow-gpu==1.14.0

$ cd PhiFlow/
$ python setup.py tf_cuda
$ pip install .
``` 
- This repository also uses [Pytorch](https://pytorch.org/) **v1.2.0** for deep learning. 
```
conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0 -c pytorch
```

## Dataset Generation
The scripts to generate different datasets would exist in `cd DataGenSctripts/`

### Single Vortex Model
```
cd DataGenScripts/
python create_dataset_single_vortex.py --domain [256, 256]
--offset [60, 60]
--n_samples 8000
--strength_range [-2, 2]
--strength_threshold 1.0
--sigma_range [5.0, 25.0]
--train_percent 0.6
--eval_percent 0.2
--num_time_steps 25 
--save_dir /path/to/save/the/dataset
```

## Neural Network Training
The scripts related to neural network training exists in `cd core/`

### Single Vortex Model
```
cd core/
python train_single_vortex.py --domain [256, 256]
--epochs 500
--dta_dir /path/to/the/dataset/
--num_time_steps 1 
--stride 1
--batch_size 32
--lr 1e-3
--l2 1e-4
--ex /name/of/the/experiment
--load_weights_ex /name/of/the/experiment/to/load/weights/to/initialize
--depth 3
--hiden_units 512
--kernel 'ExpGaussian'
```

## Visualisation

The scripts for visualisation could be found in `cd visualisations/`

- The script **vis_velocity_profiles.py** plots the variation of x-component of velocity profile along y-axis at the location of particle.
```
cd visualisations/
python vis_velocity_profiles.py --domain [256, 256]
--case_path /path/to/a/single/simualtion/case
--load_weights_ex /name/of/the/experiment/to/load/weights/to/make/predictions
--depth 3
--hiden_units 512
--num_time_steps 1 
--stride 1
--kernel 'ExpGaussian'
```
