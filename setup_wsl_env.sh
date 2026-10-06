#!/bin/bash

# default settings
ENABLE_CUDA=false
ENABLE_ROCM=false
VENV_PATH="/opt/cw-venv"

CUDA_TOOLKIT_VERSION="12-9"
ROCM_VERSION="6.4.60401-1"
ROCM_VERSION_SHORT="6.4.1"

# argument parsing
for arg in "$@"; do
  case $arg in
    --enable-cuda)
      ENABLE_CUDA=true
      shift
      ;;
	--enable-rocm)
	  ENABLE_ROCM=true
	  shift
	  ;;
	--venv-path=*)
	  VENV_PATH="${arg#*=}"
	  shift
	  ;;
    *)
      echo "Unknown option: $arg"
      exit 1
      ;;
  esac
done

# get parent directory of the script
REPO_PATH=$(dirname "$(readlink -f "$0")")


# check root privileges
if [ "$(id -u)" -ne 0 ]; then
	echo "This script must be run as root. Use 'sudo' to run it."
	exit 1
fi

# check distribution & version
if ! grep -q "Ubuntu 24.04" /etc/os-release; then
	echo "This script is intended for Ubuntu 24.04. Please run it on a compatible system."
	exit 1
fi

# Install CUDA if enabled
if [ "$ENABLE_CUDA" = true ]; then
	wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb \
		-O /tmp/cuda-keyring_1.1-1_all.deb
	dpkg -i /tmp/cuda-keyring_1.1-1_all.deb
	apt update
	apt -y install cuda-toolkit-${CUDA_TOOLKIT_VERSION} opencl-headers
fi

# Install ROCm if enabled
if [ "$ENABLE_ROCM" = true ]; then
	apt update
	wget https://repo.radeon.com/amdgpu-install/${ROCM_VERSION_SHORT}/ubuntu/noble/amdgpu-install_${ROCM_VERSION}_all.deb \
		-O /tmp/amdgpu-install_${ROCM_VERSION}_all.deb
	apt install -y /tmp/amdgpu-install_${ROCM_VERSION}_all.deb
	amdgpu-install -y --usecase=wsl,rocm --no-dkms
	apt -y install rocm-opencl-runtime
fi

# Install required packages
apt update && apt install -y \
	build-essential \
	cmake \
	git \
	python3-dev \
	python3-pip \
	python3-venv \
	python3-pybind11

# create venv
python3 -m venv $VENV_PATH
source $VENV_PATH/bin/activate

# install chipwhisperer
cd /opt
git  clone https://github.com/newaetech/chipwhisperer.git -b 5.7.0
cd chipwhisperer
git submodule update --init jupyter 
pip3 install .
pipt install -r jupyter/requirements.txt

# install plugins
cd $REPO_PATH
pip3 install pybind11
pip3 install -v .


# Clean up if any deb files were downloaded
rm -f /tmp/*.deb