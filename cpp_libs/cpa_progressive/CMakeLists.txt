#
#    Copyright (C) 2024 The University of Tokyo
#    
#    File:          /cpp_libs/cpa_progressive/CMakeLists.txt
#    Project:       sca_toolbox
#    Author:        Takuya Kojima in The University of Tokyo (tkojima@hal.ipc.i.u-tokyo.ac.jp)
#    Created Date:  23-01-2024 16:57:09
#    Last Modified: 30-01-2025 08:02:59
#

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/cpa_algorithms)

add_library(cpa_common STATIC FastCPA.cpp)
if (NOT HAVE_QUAD_PRECISION)
	target_link_libraries(cpa_common PRIVATE quadfloat leakage_model)
else()
	target_link_libraries(cpa_common PRIVATE leakage_model)
endif()

target_include_directories(cpa_common PRIVATE ${pybind11_INCLUDE_DIRS})

pybind11_add_module(cpa_kernel bind.cpp)
target_link_libraries(cpa_kernel PRIVATE cpa_common)

if (CUDA_FOUND)
	pybind11_add_module(cpa_cuda_kernel FastCPACuda.cu)
	target_link_libraries(cpa_cuda_kernel PRIVATE cpa_common)
endif()

if (OpenCL_FOUND)
	pybind11_add_module(cpa_opencl_kernel FastCPAOpenCL.cpp FastCPAOpenCLFP32.cpp)
	target_include_directories(cpa_opencl_kernel PRIVATE ${OpenCL_INCLUDE_DIRS})
	target_link_libraries(cpa_opencl_kernel PRIVATE ${OpenCL_LIBRARIES} cpa_common)
endif()

