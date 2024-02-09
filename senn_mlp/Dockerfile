# Select the base image
# FROM nvcr.io/nvidia/tensorflow:21.11-tf2-py3
FROM nvcr.io/nvidia/tensorflow:23.04-tf2-py3

# Select the working directory
WORKDIR  /senn

# Install Python requirements
COPY ./requirements.txt ./requirements.txt
RUN pip install --upgrade "jax[cuda11_local]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
RUN pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu118
RUN pip install -r requirements.txt
