#
#    Copyright (C) 2025 The University of Tokyo
#    
#    File:          /cpp_libs/model/CMakeLists.txt
#    Project:       sca_toolbox
#    Author:        Takuya Kojima in The University of Tokyo (tkojima@hal.ipc.i.u-tokyo.ac.jp)
#    Created Date:  30-01-2025 07:53:43
#    Last Modified: 30-01-2025 08:41:09
#

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/cpa_algorithms)

add_library(leakage_model STATIC AESLeakageModel.cpp)

pybind11_add_module(model_kernel bind.cpp)
target_link_libraries(model_kernel PRIVATE leakage_model)
