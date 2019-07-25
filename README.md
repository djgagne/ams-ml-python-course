# ams-ml-python-course
[![DOI](https://zenodo.org/badge/151317532.svg)](https://zenodo.org/badge/latestdoi/151317532)
Machine Learning in Python for Environmental Science Problems AMS Short Course Material

## Authors
* David John Gagne, National Center for Atmospheric Research (dgagne@ucar.edu)
* Ryan Lagerquist, University of Oklahoma (ryan.lagerquist@ou.edu)
* Greg Herman, Amazon
* Sheri Mickelson, National Center for Atmospheric Research

## Requirements
The modules for this short course require Python 3.6 and the following Python libraries:
* numpy
* scipy
* matplotlib
* xarray
* netcdf4
* pandas 
* scikit-learn
* tensorflow-gpu or tensorflow
* keras
* shapely
* descartes
* jupyter
* ipython
* jupyterlab
* ipywidgets

The current pre-compiled version of tensorflow-gpu requires your machine to have an NVIDIA GPU, CUDA 9.0, CUDA Toolkit 9.0, and cuDNN 7. If you have different versions of CUDA available, you will have to build Tensorflow from source, which can take a few hours.

GPUs are recommended for modules 3 and 4 but are not needed for modules 1 and 2.

## Data Access
The data for the course are stored online. The `download_data.py` script will download the data to the appropriate location and extract all files. The netCDF data is contained in a 2GB tar file, so make sure you have at least 4GB of storage available and a fast internet connection.

## Course Videos
* [Module 1: Data Analysis and Pre-Processing](https://drive.google.com/open?id=1o1R-UinPjxn-gpNTtNloB6mn9Gus1th8)
* [Module 2: Machine Learning with Scikit-Learn](https://drive.google.com/open?id=1WUk4lZBTSOE_kf9_1JQDOoMAOFi6b698)
* [Module 3: Deep Learning with Keras](https://drive.google.com/open?id=1tOPGC71Yx8ygvnRzws5KLuRW20-bSTq8)
* [Module 4: Model Interpretation](https://drive.google.com/open?id=1NUxdreKkUXAevZhT0eSJTm5Z2JuqV3Ry)

## Setup Instructions (Local Install; CPU Only)
These instructions assume you have a bash shell running or the Windows command prompt. Conda environments do not work in csh. 
1. Install the [miniconda](https://conda.io/miniconda.html) Python distribution.
2. Create a separate conda environment for the short course: `conda create -n mlpy python=3.6`
3. Activate the enviornment by running `source activate mlpy` (bash in linux or mac) or `activate mlpy` (Windows)
4. Install the required base libraries: `conda install pip numpy scipy matplotlib scikit-learn netcdf4 xarray pandas ipython jupyter ipywidgets shapely descartes`
5. Install tensorflow and keras: `pip install tensorflow; pip install keras`
6. Clone the short course repository: `git clone https://github.com/djgagne/ams-ml-python-course.git`
7. Change into the ams-ml-python-course directory.
8. Download the course data to your local machine: `python download_data.py`
9. Start Jupyter lab: `jupyter lab`
10. Each module is in a separate folder. Open the Jupyter notebook in each folder and follow instructions. If you have problems, please create an issue on the Github repository site.

## Setup Instructions (Docker)
These instructions are for those who want to run the short course Docker image either on their local machine (requires Docker to be installed) or on a single cloud VM. 
1. Install [Docker](https://www.docker.com/get-started).
2. From the command line, pull the appropriate short course Docker container:
    * CPU only: `docker pull djgagne/ams-ml-python-course:cpu`
    * GPU (requires NVIDIA GPU, CUDA and nvidia-docker): `docker pull djgagne/ams-ml-python-course:gpu`
3. To start the container: `docker run -p 8888:8888 djgagne/ams-ml-python-course:cpu` or `:gpu` if you are using the CPU or GPU version.
4. To access jupyter lab, open a web browser to localhost:8888 and paste in the token string from the command line.
5. If you are running on a remote server, you will need to forward port 8888 to your local machine. You can do this over ssh if it is a remote server or through the web if you are running on a cloud server with port 8888 opened.

# **Optional** Setting up GPU-enabled short course Jupyter hub containers
These instructions are for creating and managing your own short course managed by Jupyterhub on Kubernetes with everything in a Docker container. You do not need to follow these instructions if you are just trying to run the short course modules locally.
## Requirements for architecture
* Docker
* Google Compute Engine
* Google Kubernetes Engine
* NVIDIA CUDA docker [images](https://hub.docker.com/r/nvidia/cuda)
* jupyter [docker-stacks](https://github.com/jupyter/docker-stacks)

## Recipe
* Start a Google Compute Engine instance with an NVIDIA GPU and install CUDA and docker. See [here](https://medium.com/google-cloud/jupyter-tensorflow-nvidia-gpu-docker-google-compute-engine-4a146f085f17).
* Clone the jupyter docker-stacks repository
* In the base-notebook Docker file, change the `BASE_CONTAINER` to "nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04"
* Build base notebook: `>> docker build --rm -t username/base-notebook .`
* Change to docker-stacks/minimal-notebook directory and change the FROM option to username/base-notebook.
* Build minimal notebook `>> docker build --rm -t username/minimal-notebook .`
* Change to directory containing short course docker file.
* Build the short course container `>> docker build --rm -t username/ams-ml-short-course:gpu .`
* Login to docker hub with `>> docker login`
* Push your container to Docker Hub.
* Start a Kubernetes cluster on Google Cloud with 1 CPU node and 1 GPU node. Use preemptible instances to save a lot of money.
* Log into a Kubernetes node and install CUDA [here](https://cloud.google.com/compute/docs/gpus/add-gpus).
* Wait until the nvidia drivers have been completely installed. Check status by typing in
`kubectl get pods --all-namespaces` and wait for everything to be running.
* Setup Jupyterhub on Google Cloud by following instructions [here](https://zero-to-jupyterhub.readthedocs.io/en/stable/index.html).

