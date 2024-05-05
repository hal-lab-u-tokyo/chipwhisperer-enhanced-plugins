# Chipwhisperer-Enhanced-Plugins
This repository contains enhanced plugins for Chipwhisperer.
The plugins are designed to be used with the Chipwhisperer platform.

## Extended features
* Target devices (See [hardware](docs/hardware.md) for details)
  * SAKURA-X board
  * ESP32
* Capture devices
  * Keysight Infiniviion 40000 (MSO-X 4101A tested)
  * Rigol MSO8000 (MSO8104 tested)
* Analysis algorithm
  * Fast correlation power analysis (FastCPA) with
    * OpenMP parallelization
	* CUDA for NVIDIA GPU
    * OpenCL for AMD, Intel, Apple GPUs
  * Quad-precision floating-point emulation for double-precision-only CPU (e.g, Apple Silicon CPUs)
  * Double-precision floating-point emulation in OpenCL for single-precision-only GPUs (e.g., Apple Silicon GPUs, Intel Arc GPUs)

## Contents
* lib: python library source
* notebooks: Jupyter Notebook files as examples
* udev-rules: udev rule file for devices related to side-channel attack evaluation
* docs: documents
* cpp_libs: C++ library source
* docker: Dockerfile for building a docker image

## Setup guides
### Capturing traces
See [Setup](docs/setup.md)
### Analysis
It is recommended that the Docker image be used with docmer-compose.

See [docker](docs/docker.md)

For MacOS users, [install_mac.sh](./install_mac.sh) is provided to install this repository and its dependencies.

## Getting Started
### Acquiring traces
[acquire_traces](notebooks/acquire_traces.ipynb) is a sample notebook for acquiring traces using VISA compatible oscilloscopes.

### Analysis with FastCPA
[fastcpa_example](notebooks/fastcpa_example.ipynb) is a sample notebook for analyzing traces using FastCPA.

#### OpenCL device selection
If you have multiple OpenCL devices, you can specify the device to use by setting the `CL_PLATFORM` and `CL_DEVICE` environment variables.

# Known Issues
The OpenCL implementation of FastCPA in Apple Silicon GPUs may provide incorrect results when a large number of traces and a long waveform are used.
In this case, the update interval should be set to a smaller value.

## Special Thanks to Those Who Assisted in this development
* Masaki Morita
* Youhyun Kim
