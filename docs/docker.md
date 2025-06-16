First, ensure you have Docker installed on your system. You can find installation instructions on the [Docker website](https://docs.docker.com/get-docker/).
As recommended, it is better to docker compose (v2) integrated with Docker CLI to run the container, instead of using the legacy docker-compose command.

# Docker Compose File Variants

This repository includes multiple Docker Compose files tailored for different platforms. By overriding the default `docker-compose.yml` file with one of the files listed below, you can build and run containers optimized for your desired platform.

All the following commands are executed at the root of the repository.

# Building the Docker Image
## OpenMP enabled image (default)
```bash
docker compose build
```

## CUDA enabled image
```bash
docker compose -f docker-compose.yml -f docker/docker-compose.override.nvidia.yml build
```

## OpenCL enabled image for AMDGPU
```bash
docker compose -f docker-compose.yml -f docker/docker-compose.override.amdgpu.yml build
```

## OpenCL enabled image for Intel GPU
```bash
docker compose -f docker-compose.yml -f docker/docker-compose.override.intel-gpu.yml build
```

# Running the Docker Container
The container launches a Jupyter Notebook server.

## Common options
Start containers with the following environment variables:
* JUPYTER_PORT: Port number for Jupyter Notebook (default: `8888`)
* JUPYTER_TOKEN: Token for Jupyter Notebook (default: `chipwhisperer`)
* NOTEBOOK_DIR: Directory to mount as a volume in the container (default: `notebooks`)
* DATA_DIR: Directory to mount as a volume in the container (default: `data`)

## Start container
At the root of the repository, run the following command to start the container in detached mode:
```bash
sh launch_docker.sh [TYPE] [--as-root]
```
The `TYPE` argument can be one of the following:
* `cpu`: OpenMP enabled container (default)
* `cuda`: CUDA enabled container
* `amdgpu`: OpenCL enabled container for AMDGPU
* `intel-gpu`: OpenCL enabled container for Intel GPU

The `--as-root` option allows you to run the jupyter notebook server as root in the container.
Some GPU environments may require this option to be set, depending on the host accout permissions.
However, created files on the mounted volume will be owned by root, which may cause permission issues when accessing them from the host.

## Note for CUDA enabled container

If you want to change used GPU in multi-GPU environment, modify `device_ids` item in `docker-compose.override.nvidia.yml`.

## Note for OpenCL enabled container for AMDGPU

If you want to specify the OpenCL platform and device ID, add the following environment variables to the `docker-compose.override.amdgpu.yml` file:
* OPENCL_PLATFORM: OpenCL platform ID (default: 0)
* OPENCL_DEVICE: OpenCL device ID (default: 0)

## Note for OpenCL enabled container for Intel GPU
The OpenCL platform and device ID can be specified in the same way as for AMDGPU.

# Connecting to the container
## Jupyter Notebook
After starting the container, you can access the Jupyter Notebook server by opening a web browser and navigating to:
```
http://localhost:<JUPYTER_PORT>/?token=<JUPYTER_TOKEN>
```
or from other machines:
```
http://<HOST_IP>:<JUPYTER_PORT>/?token=<JUPYTER_TOKEN>
```
Please replace `<JUPYTER_PORT>` with the port number you specified in the `JUPYTER_PORT` environment variable as described [here](#common-options).
If default value is used, http://localhost:8888/?token=chipwhisperer will be used.


## Command line Interface (shell)
The above command will start the container "chipwhisperer-notebook" in detached mode.
Other than jupyter notebook, you can also run the following command to start a bash shell in the container:


```bash
docker compose exec -it notebook /bin/bash
```
or
```bash
docker exec -it chipwhisperer-notebook /bin/bash
```
