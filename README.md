# Self-Expanding Neural Networks

This repository contains all code necessary to reproduce our experiments from the main body of the paper "Self-Expanding
Neural Networks". There are two codebases, one for CNN experiments and one for MLP experiments. 
They are located in the senn_cnn and senn_mlp folders respectively.

## Setting up the CNN Codebase
Setting up the CNN codebase requires to navigate to the senn_cnn and proceed with the requirements and environment variables as described below.

### Installing Requirements
The requirements can be installed using PDM. To do so, first install PDM, then run ```pdm install``` to install the requirements. This will setup a virtual environment with all dependencies. We provide a Dockerfile as well which should be used after setting up pdm.

#### Docker
In order ease reproduction, we provide a Dockerfile.
We also provide a docker-compose service to setup the docker container.
To use Docker, first install Docker and docker-compose, then run ```docker compose build``` to build the container.
Then run ```docker compose up -d``` to start the container. This will start a container with all dependencies installed.

#### PDM
It is possible to install the requirements using PDM, a python package manager. To do so, first install PDM, then run
```pdm install``` to install the requirements. This will setup a virtual environment with all dependencies.

### Environment Variables
The following environment variables are used:
CUDA_VISIBLE_DEVICES: Specifies which GPU to use.
DATASETS_ROOT_DIR: Specifies the root directory for datasets.
WANDB_DIR: Specifies the root directory for wandb logs.
RTPT_INITIALS: Specifies the initials to use for RTPT.
WANDB_API_KEY: Specifies the API key to use for wandb. This is only necessary if you want to log to wandb.
If using Docker, these may be set in the docker-compose.yml file, or by creating a .env file in the root directory.
If using a .env file create it before running ```docker compose up -d```.

### Running Experiments
To run an experiment, simply run the corresponding script. If using Docker, this may be done by running
```docker compose exec -e CUDA_VISIBLE_DEVICES -e JAX_PLATFORMS main python experiments/{experiment}.py```
where experiment is on of:
- senn_cifar10_manycycle
- senn_cifar10_onecycle
- transfer_cifar10_pretrain
- transfer_tinyimagenet_fixed
- transfer_tinyimagenet_senn
If using pdm only, source the virtual environment, then run ```python experiments/{experiment}.py```.

#### Transfer Learning Experiments
The transfer learning experiments save a model checkpoint after training. This checkpoint is then loaded and used to
initialize the model for the transfer learning task. In docker the directory is /senn/orbax/pretrained/final. If you want to alter the checkpoint dir you can change the CHECKPOINT_DIR variable in each experiment file.

#### Experiment Settings
The hyperparameters and tasks may be varied by editing the settings in the corresponding experiment file.

#### Changing the random seed
The random seeds used for the model initialization training can be altered in the experiment script.

## Setting up the MLP Codebase
You can setup the MLP codebase by navigating to the senn_mlp folder and proceeding with the requirements and environment variables as described below.
### Installing Requirements
We provide two methods to quickly install all dependencies required for our experiments.

#### Docker
In order ease reproduction, we provide a Dockerfile and scripts with which to use it.
1. To build the container run ```sh build.sh``` in the main ("senn") directory.
2. In order to run the container on gpu 0 execute the command ```./use 0```, where this argument may be omitted if using cpu.
3. To start a container with port forwarding for tensorboard use ```sh tboard.sh```

### Pip
It is of course possible to install dependencies with pip, without using Docker:
1. Install JAX - follow instructions in [official repository](https://github.com/google/jax), e.g. for cuda as of May 2023: 
```pip install --upgrade "jax[cuda11_local]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html```.
2. Install PyTorch and Torchvision - instructions at [official website](https://pytorch.org/get-started/locally/), e.g. as of May 2023:
```pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu118```. Make sure to install a version with CUDA version requirements compatible with JAX.
3. Install other requirements from requirements.txt:
```pip install -r requirements.txt```

### Running Experiments
If you are not using Docker, adjust the "checkpoints", "datasets" and "logs" folders in config.yaml to your liking. To
run an experiment, simply execute the corresponding script, e.g. for experiment 4:
```python experiment4.py --name my_experiment```.
The "--name my_experiment" argument specifies that tensorboard will store any logs under that name.

#### Tensorboard Results
Various metrics, such as training/validation loss and neuron count will be logged during training. If using
Docker, these metrics may be viewed by starting a container with ```sh tboard.sh``` and then running the command
```tensorboard --logdir /logs``` inside it. If Docker is not used, then simply run ```tensorboard --logdir ./mylogs```,
replacing "./mylogs" with the directory chosen in "config.yaml" for "meta:logdir".

#### Experiment Settings
The hyperparameters and tasks may be adjusted in the "default_config.yaml" in the folder corresponding to the experiment
e.g. for experiment 4 one would adjust "senn/experiment4/default_config.yaml".

##### Changing the random seed
The random seed used during an experiment may be varied by altering "meta:seed", or by passing the commandline argument
```--seed```, e.g. ```python experiment4.py --name my_experiment --seed 2```.

##### Subset size in experiment 4
In experiment 4 there is an additional replication relevant setting, "data:defaults:N", initially set to 4800. This
specifies the number of examples from each class which will be included in the subset of MNIST trained on. For example,
N=6000 would result in training on the full 60000 images.
