/*
*    Copyright (C) 2025 The University of Tokyo
*    
*    File:          /cpp_libs/socpa/bind.cpp
*    Project:       sca_toolbox
*    Author:        Takuya Kojima in The University of Tokyo (tkojima@hal.ipc.i.u-tokyo.ac.jp)
*    Created Date:  30-01-2025 08:00:08
*    Last Modified: 30-01-2025 09:04:41
*/


#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "SOCPA.hpp"

#include "AESLeakageModel.hpp"

PYBIND11_MODULE(socpa_kernel, module) {
	module.doc() = "2nd-Order CPA"; // optional module docstring

	py::class_<ProductCombineSOCPA>(module, "ProductCombineSOCPA")
		.def(py::init<int, int, int, AESLeakageModel::ModelBase*, py::array_t<TRACE_T> &>())
		.def("calculate_correlation", &ProductCombineSOCPA::calculate_correlation);
}
