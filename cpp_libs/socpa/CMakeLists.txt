#
#    Copyright (C) 2024 The University of Tokyo
#    
#    File:          /cpp_libs/socpa/CMakeLists.txt
#    Project:       sca_toolbox
#    Author:        Takuya Kojima in The University of Tokyo (tkojima@hal.ipc.i.u-tokyo.ac.jp)
#    Created Date:  23-01-2024 16:57:09
#    Last Modified: 06-05-2025 06:08:01
#

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/cpa_algorithms)

add_library(socpa_common STATIC SOCPA.cpp)
if (NOT HAVE_QUAD_PRECISION)
	target_link_libraries(socpa_common PRIVATE quadfloat leakage_model)
else()
	target_link_libraries(socpa_common PRIVATE leakage_model)
endif()

target_include_directories(socpa_common PRIVATE ${pybind11_INCLUDE_DIRS})

pybind11_add_module(socpa_kernel bind.cpp)
target_link_libraries(socpa_kernel PRIVATE socpa_common)

if (CUDA_FOUND)
	pybind11_add_module(socpa_cuda_kernel SOCPACuda.cu)
	target_link_libraries(socpa_cuda_kernel PRIVATE socpa_common)
endif()

if (OpenCL_FOUND)
	pybind11_add_module(socpa_opencl_kernel SOCPAOpenCL.cpp SOCPAOpenCLFP32.cpp)
	target_include_directories(socpa_opencl_kernel PRIVATE ${OpenCL_INCLUDE_DIRS})
	target_link_libraries(socpa_opencl_kernel PRIVATE ${OpenCL_LIBRARIES} socpa_common)
endif()

