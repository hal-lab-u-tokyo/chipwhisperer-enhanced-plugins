# Chipwhisperer-Enahnced-Plugins
This repository contains enhanced plugins for Chipwhisperer.
The plugins are designed to be used with the Chipwhisperer platform.

## Extended features
* Target device
  * SAKURA-X board
  * ESP32
* Capture device
  * Keysight Infiniviion 40000 (MSO-X 4101A tested)
  * Rigol MSO8000 (MSO8104 tested)
* Analysis algorithm
  * Fast correlation power analysis (FastCPA) with
    * OpenMP parallelization
	* CUDA for NVIDIA GPU
    * OpenCL for AMD, Intel, Apple GPUs

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
Recommended to use the Docker image with docmer-compose.

See [docker](docs/docker.md)

For MacOS users, `install_mac.sh` is provided to install this repository and its dependencies.

## Special Thanks to Those Who Assisted in this development
* Masaki Morita
* Youhyun Kim