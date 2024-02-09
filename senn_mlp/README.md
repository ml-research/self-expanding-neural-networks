# Self-Expanding Neural Networks

This repository contains all code necessary to reproduce our experiments from the main body of the paper "Self-Expanding
Neural Networks". There are four scripts, "experiment{N}.py", corresponding to the four experiments ordered by appearance in the main body.
That is, least squares regression, classification of in 2D, image classification, and variation of converged width with dataset size.

## Installing Requirements
We provide two methods to quickly install all dependencies required for our experiments.

### Docker
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

## Running Experiments
If you are not using Docker, adjust the "checkpoints", "datasets" and "logs" folders in config.yaml to your liking. To
run an experiment, simply execute the corresponding script, e.g. for experiment 4:
```python experiment4.py --name my_experiment```.
The "--name my_experiment" argument specifies that tensorboard will store any logs under that name.

### Tensorboard Results
Various metrics, such as training/validation loss and neuron count will be logged during training. If using
Docker, these metrics may be viewed by starting a container with ```sh tboard.sh``` and then running the command
```tensorboard --logdir /logs``` inside it. If Docker is not used, then simply run ```tensorboard --logdir ./mylogs```,
replacing "./mylogs" with the directory chosen in "config.yaml" for "meta:logdir".

### Experiment Settings
The hyperparameters and tasks may be adjusted in the "default_config.yaml" in the folder corresponding to the experiment
e.g. for experiment 4 one would adjust "senn/experiment4/default_config.yaml".

#### Changing the random seed
The random seed used during an experiment may be varied by altering "meta:seed", or by passing the commandline argument
```--seed```, e.g. ```python experiment4.py --name my_experiment --seed 2```.

#### Subset size in experiment 4
In experiment 4 there is an additional replication relevant setting, "data:defaults:N", initially set to 4800. This
specifies the number of examples from each class which will be included in the subset of MNIST trained on. For example,
N=6000 would result in training on the full 60000 images.
