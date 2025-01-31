#
#    Copyright (C) 2024 The University of Tokyo
#    
#    File:          /cpp_libs/CMakeLists.txt
#    Project:       tkojima
#    Author:        Takuya Kojima in The University of Tokyo (tkojima@hal.ipc.i.u-tokyo.ac.jp)
#    Created Date:  16-01-2024 21:01:52
#    Last Modified: 30-01-2025 08:06:02
#

cmake_minimum_required(VERSION 3.20)

project(cw_enhanced_plugin_cpp_libs VERSION 0.1.0)
set(Python_VIRTUALENV FIRST)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

# add include directory globally
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# build type and optimization flags
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# find the pybind11 package for giving hints
execute_process(
  COMMAND
	${Python3_EXECUTABLE}  -c "import os; import pybind11; print(os.path.dirname(pybind11.__file__))"
  OUTPUT_VARIABLE
    PYBIND11_SITE_PACKAGES
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_package(pybind11 REQUIRED HINTS ${PYBIND11_SITE_PACKAGES})
set(CMAKE_CXX_STANDARD 11)

# check if quad precision is supported on the system
include(./cmake/quadfloat.cmake)
if (NOT ${HAVE_QUAD_PRECISION})
	add_subdirectory(soft_quadfloat)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSOFT_QUAD_PRECISION")
endif()

# find OpenMP support for MacOS
if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
	execute_process(COMMAND brew --prefix libomp
					OUTPUT_VARIABLE OpenMP_ROOT
					OUTPUT_STRIP_TRAILING_WHITESPACE)  

	set(OpenMP_C_LIB_NAMES "omp")
	set(OpenMP_CXX_LIB_NAMES "omp")
	set(OpenMP_omp_LIBRARY "${OpenMP_ROOT}/lib/")

	include_directories("${OpenMP_ROOT}/include")
	link_directories("${OpenMP_ROOT}/lib/")
	set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -lomp")

endif()

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
		if (NOT ${HAVE_QUAD_PRECISION})
			set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DSOFT_QUAD_PRECISION")
		endif()
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

add_subdirectory(model)
add_subdirectory(cpa_progressive)
add_subdirectory(socpa)