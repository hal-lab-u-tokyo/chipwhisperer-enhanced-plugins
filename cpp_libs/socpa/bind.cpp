/*
*    Copyright (C) 2025 The University of Tokyo
*    
*    File:          /cpp_libs/socpa/bind.cpp
*    Project:       sca_toolbox
*    Author:        Takuya Kojima in The University of Tokyo (tkojima@hal.ipc.i.u-tokyo.ac.jp)
*    Created Date:  30-01-2025 08:00:08
*    Last Modified: 01-05-2025 05:46:46
*/


#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "SOCPA.hpp"

#include "AESLeakageModel.hpp"

PYBIND11_MODULE(socpa_kernel, module) {
	module.doc() = "2nd-Order CPA"; // optional module docstring

	py::class_<SOCPA>(module, "SOCPA")
		.def(py::init<int, int, int, AESLeakageModel::ModelBase*>())
		.def("calculate_correlation", &SOCPA::calculate_correlation)
		.def("get_max_combined_offset", &SOCPA::get_max_combined_offset)
		.def("set_point_tile_size", &SOCPA::set_point_tile_size)
		.def("set_trace_tile_size", &SOCPA::set_trace_tile_size);

}
