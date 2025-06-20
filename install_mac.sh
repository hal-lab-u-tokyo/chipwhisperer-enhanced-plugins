#!/bin/sh

NO_CPP_LIB=0

# check python3 is installed
echo "-- Checking if python3 is installed."
if test ! $(which python3 2> /dev/null); then
	echo "Fatal: python3 is essential for this framework."
	exit 1;
fi

# check xcode command line tools are installed
echo "-- Checking if xcode command line tools are installed."
if test ! $(which xcode-select 2> /dev/null); then
	printf "-- xcode command line tools is needed to build C++ acceleration libraries.\nWould you like to proceed without installing xcode command line tools? (y/n):"
	read answer
	if [ "$answer" != "${answer#[Yy]}" ] ;then
		echo "Proceeding without xcode command line tools."
		NO_CPP_LIB=1
	else
		exit 1;
	fi
fi


# check if brew is installed
NO_BREW=0
echo "-- Checking if homebrew is installed."
if test ! $(which brew 2> /dev/null); then
	printf "homebrew is need to install OpenMP.\nWould you like to proceed without installing homebrew? (y/n):"
	read answer
	if [ "$answer" != "${answer#[Yy]}" ] ;then
		echo "Proceeding without homebrew."
		NO_BREW=1
	else
		exit 1;
	fi
else
	brew install libomp
fi

# check if cmake is installed
echo "-- Checking if cmake is installed."
if test ! $(which cmake 2> /dev/null); then
	printf "-- cmake is needed to build C++ acceleration libraries.\nWould you like proceed without installing cmake? (y/n):"
	read answer
	if [ "$answer" != "${answer#[Yy]}" ] ;then
		echo "Proceeding without cmake."
		NO_CPP_LIB=1
	else
		if [ $NO_BREW -eq 0 ]; then
			printf "Would you like to install cmake? (y/n):"
			read answer
			if [ "$answer" == "${answer#[Yy]}" ] ;then
				brew install cmake
			else
				echo "Quitting installation."
				exit 1;
			fi
		else
			echo "Fatal: cmake is essential to build C++ acceleration libraries."
			exit 1;
		fi
	fi
fi

# prepare venv
echo "-- Preparing virtual environment."
# ask where to install the virtual environment
printf "Where would you like to install the virtual environment? (default: ./venv):"
read venv_path
if [ -z "$venv_path" ]; then
	venv_path="./venv"
fi
# create the virtual environment
if [ -d "$venv_path" ]; then
	printf "The directory $venv_path already exists. Would you like to use existing environment? (y/n):"
	read answer
	if [ "$answer" == "${answer#[Yy]}" ] ;then
		exit 1;
	fi
else
	python3 -m venv $venv_path
fi
source $venv_path/bin/activate
work_dir=`pwd`

# clone chipwhisperer in temp directory
echo "-- Cloning chipwhisperer."
temp_dir_name=$(mktemp -d -t chipwhisperer-XXXXXXXXXX)
git clone https://github.com/newaetech/chipwhisperer.git $temp_dir_name
cd $temp_dir_name
git checkout 5.7.0
git submodule update --init jupyter
pip3 install .
pip3 install -r jupyter/requirements.txt
rm -rf $temp_dir_name

cd $work_dir

if [ $NO_CPP_LIB -eq 0 ]; then
	echo "-- Installing pybind11."
	pip3 install pybind11
fi

echo "-- Updating pip, setuptools and wheel."
pip3 install --upgrade pip setuptools wheel

echo "-- Installing chipwhisperer-enhanced-plugins."
pip3 install . -v

echo "-- Installation sucessfully completed."
