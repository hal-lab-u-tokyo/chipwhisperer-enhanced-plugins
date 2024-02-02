# Build image
## OpenMP enabled image
```bash
docker-compose
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

## Start OpenCL enabled container for AMDGPU
```bash
docker-compose -f docker-compose.yml -f docker/docker-compose.override.amdgpu.yml up -d
```

## Start OpenCL enabled container for Intel GPU
```bash
docker-compose -f docker-compose.yml -f docker/docker-compose.override.intel-gpu.yml up -d
```




