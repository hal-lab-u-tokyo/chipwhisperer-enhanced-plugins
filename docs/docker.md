# Build image
Execute the following command at the root of the repository:
## OpenMP enabled image
```bash
docker-compose build
```

## CUDA enabled image
```bash
docker-compose -f docker-compose.yml -f docker/docker-compose.override.nvidia.yml build
```

## OpenCL enabled image for AMDGPU
```bash
docker-compose -f docker-compose.yml -f docker/docker-compose.override.amdgpu.yml build
```

## OpenCL enabled image for Intel GPU
```bash
docker-compose -f docker-compose.yml -f docker/docker-compose.override.intel-gpu.yml build
```

# Run container
The container launches a Jupyter Notebook server.

## Common options
Start containers with the following environment variables:
* JUPYTER_PORT: Port number for Jupyter Notebook
* JUPYTER_TOKEN: Token for Jupyter Notebook
* NOTEBOOK_DIR: Directory to mount as a volume in the container
* DATA_DIR: Directory to mount as a volume in the container

## Start OpenMP enabled container
```bash
docker-compose up -d
```

## Start CUDA enabled container
```bash
docker-compose -f docker-compose.yml -f docker/docker-compose.override.nvidia.yml up -d
```
If you want to change used GPU in multi-GPU environment, modify `device_ids` item in `docker-compose.override.nvidia.yml`.

## Start OpenCL enabled container for AMDGPU
```bash
docker-compose -f docker-compose.yml -f docker/docker-compose.override.amdgpu.yml up -d
```
If you want to specify the OpenCL platform and device ID, add the following environment variables to the `docker-compose.override.amdgpu.yml` file:
* OPENCL_PLATFORM: OpenCL platform ID (default: 0)
* OPENCL_DEVICE: OpenCL device ID (default: 0)

## Start OpenCL enabled container for Intel GPU
```bash
docker-compose -f docker-compose.yml -f docker/docker-compose.override.intel-gpu.yml up -d
```

The OpenCL platform and device ID can be specified in the same way as for AMDGPU.

