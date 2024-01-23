#
#    Copyright (C) 2024 The University of Tokyo
#    
#    File:          /cpp_libs/CMakeLists.txt
#    Project:       tkojima
#    Author:        Takuya Kojima in The University of Tokyo (tkojima@hal.ipc.i.u-tokyo.ac.jp)
#    Created Date:  16-01-2024 21:01:52
#    Last Modified: 22-01-2024 14:09:26
#

cmake_minimum_required(VERSION 3.20)

project(pybind_kernel VERSION 0.1.0)
set(Python_VIRTUALENV FIRST)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

# find the pybind11 package for giving hints
execute_process(
  COMMAND
	${Python3_EXECUTABLE}  -c "import os; import pybind11; print(os.path.dirname(pybind11.__file__))"
  OUTPUT_VARIABLE
    PYBIND11_SITE_PACKAGES
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_package(pybind11 REQUIRED HINTS ${PYBIND11_SITE_PACKAGES})

find_package(OpenMP)
find_package(CUDA)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(OpenMP_FOUND)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()


if (CUDA_FOUND)
	# automatically add the CUDA bin directory to the path if it is not already included
	set(ENV_PATH "$ENV{PATH}")
	string(FIND "${ENV_PATH}" "${CUDA_TOOLKIT_ROOT_DIR}/bin" cuda_bin_pos)
	if(cuda_bin_pos EQUAL -1)
		set(ENV{PATH} "${CUDA_TOOLKIT_ROOT_DIR}/bin:${ENV_PATH}")
	endif()
	enable_language(CUDA)
	if (OpenMP_FOUND)
		set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${OpenMP_CXX_FLAGS}")
	endif()
else()
	message(STATUS "Skip building CUDA libraries")
endif()

# if env OpenCL_LIBRARY is set, use it
if (DEFINED ENV{OpenCL_LIBRARY})
	set(OpenCL_LIBRARY $ENV{OpenCL_LIBRARY})
	message(STATUS "OpenCL_LIBRARY is set to $ENV{OpenCL_LIBRARY}")
endif()
find_package(OpenCL)
if (OpenCL_FOUND)
	message(STATUS "OpenCL found")
else()
	message(STATUS "Skip building OpenCL libraries")
endif()

add_subdirectory(cpa_progressive)
