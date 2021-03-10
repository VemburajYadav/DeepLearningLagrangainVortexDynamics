# Simulation of Fluid Flows based on Data-driven Evolution of Vortex Particles

This repository contains the code for my master thesis project for the masters programme Computational Sciences in Engineering (CSE) at TU Braunschweig.  
- [Report](https://www.dropbox.com/s/jrbpttt3a17pf1s/Report_revised_4868459.pdf?dl=0) for the project.
- [Slides](https://www.dropbox.com/scl/fi/pyqj9hwrk6370iqmfdo0d/Presentation_MA.pptx?dl=0&rlkey=aywt3nogtie7ni28brnrjr19b) of the final presentation.


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

## Neural Networks for Vortex Particle Dynamics (Open domain)

In order to generate datasets for training neural networks to predict dynamics of vortex particles, we use [PhiFlow](https://github.com/tum-pbs/PhiFlow')
to perform numerical simulations to create data samples with grid-based velocity fields. For more information related to our process of generating data samples, and an additional optimization routine to obtain pseudo-labels for features of vortex particles form the grid-based velocity fields, which we refer to as 
**Vortex-Fit**, please refer to Section 3.2 of the [Report](https://www.dropbox.com/s/jrbpttt3a17pf1s/Report_revised_4868459.pdf?dl=0)
   
The scripts to generate different datasets exist in `cd DataGenSctripts/`.

### Datasets

- To create dataset for **inviscid** flows, execute the following script
```
cd DataGenScripts/
python create_dataset_dataset_multi_vortex.py --domain [120, 120]
--offset [40, 40]
--n_samples 4000
--sigma_range [2.0, 10.0]
--train_percent 0.6
--eval_percent 0.2
--num_time_steps 10
--time_step 0.2 
--save_dir '/path/to/save/the/dataset'
```

- To create dataset for **viscous** flows, execute the following script
```
cd DataGenScripts/
python create_dataset_dataset_multi_vortex.py --domain [120, 120]
--offset [40, 40]
--n_samples 4000
--n_particles 10
--sigma_range [2.0, 10.0]
--viscosity_range [0.0, 3.0]
--train_percent 0.6
--eval_percent 0.2
--num_time_steps 10
--time_step 0.2 
--save_dir '/path/to/save/the/dataset'
```

- The values given above to different input arguments for the running the script are the default values used in our work for which we present the results.
Only change the `--save_dir` argument to specify the path to the directory, where the dataset needs to be saved. 

- In the path specified by `--save_dir`, three different sub-directories named **train**, **val** and **test**
will be created, where the train-val-test split is controlled by the `-train_percent` and `--eval_percent` arguments to the script. The overall number number of data samples is controlle by the `--n_samples` argument.

- We consider a domain with 120 x 120 grid cells of unit length, which is specified by the `--domain` argument as a list.

- The argument `--n_particles` specifies the number of vortex particles 
- The argument `--sigma_range` specifies the range of values for **core size** of vortex particles, from which the values are uniformly sampled from. Same applies for the `--viscosity` argument in the script for viscous flows.
 

### Neural Network Training
The scripts related to neural network training and evaluation exists in `cd core/`

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
